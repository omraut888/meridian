"""Grounded answer generation.

The pipeline depends only on :class:`AnswerGenerator` and the provider-neutral
event types below; :class:`ClaudeGenerator` is the current implementation.
Retrieved chunks are passed to Claude as ``search_result`` content blocks with
citations enabled, so every citation maps back to an exact retrieved chunk
rather than to a free-text reference the model could invent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import structlog
from anthropic.types.beta import BetaSearchResultBlockParam

from meridian.config import GenerationSettings
from meridian.exceptions import GenerationError
from meridian.models import ScoredChunk

log = structlog.get_logger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You answer questions using only the search results provided in the user turn.

Ground every factual claim in those results; the citation system attaches \
sources to your sentences automatically when you draw on a result. If the \
results do not contain enough information to answer, say so plainly and state \
what is missing rather than filling gaps from general knowledge. When results \
disagree, say that they disagree and summarize each position.

Write for a technically fluent reader: direct, specific, no preamble."""


@dataclass(frozen=True, slots=True)
class AnswerText:
    """An incremental fragment of answer text."""

    text: str


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    """A citation attached to the preceding answer text.

    Attributes:
        source_index: Index into the chunks passed to ``stream_answer``.
        chunk_id: ID of the cited chunk.
        cited_text: The exact text span of the chunk that was cited.
    """

    source_index: int
    chunk_id: str
    cited_text: str


@dataclass(frozen=True, slots=True)
class AnswerComplete:
    """Terminal event for a generation stream.

    Attributes:
        stop_reason: Provider stop reason (``"end_turn"``, ``"max_tokens"``, ``"refusal"``, ...).
        model: The model that actually produced the answer (differs from the
            requested model if a server-side fallback ran).
        input_tokens: Billed input tokens.
        output_tokens: Billed output tokens.
        refused: True if the answer was declined; any streamed text should be discarded.
    """

    stop_reason: str | None
    model: str
    input_tokens: int
    output_tokens: int
    refused: bool


AnswerEvent = AnswerText | AnswerCitation | AnswerComplete


class AnswerGenerator(Protocol):
    """Provider-agnostic streaming answer generator."""

    def stream_answer(self, question: str, sources: Sequence[ScoredChunk]) -> AsyncIterator[AnswerEvent]:
        """Stream an answer to ``question`` grounded in ``sources``."""
        ...


class ClaudeGenerator:
    """:class:`AnswerGenerator` backed by the Anthropic Messages API."""

    def __init__(self, settings: GenerationSettings) -> None:
        """Create the generator.

        Credentials resolve from ``MERIDIAN_GENERATION__API_KEY`` if set,
        otherwise from the SDK's standard chain (``ANTHROPIC_API_KEY``, an
        ``ant auth login`` profile, ...).
        """
        self._settings = settings
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            timeout=settings.timeout_s,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def stream_answer(
        self, question: str, sources: Sequence[ScoredChunk]
    ) -> AsyncIterator[AnswerEvent]:
        """Stream a cited answer.

        Args:
            question: The user's question.
            sources: Retrieved chunks, in rank order.

        Yields:
            :class:`AnswerText` and :class:`AnswerCitation` events, then exactly
            one :class:`AnswerComplete`.

        Raises:
            GenerationError: If the API call fails (after the SDK's own retries).
        """
        request: dict[str, Any] = {
            "model": self._settings.model,
            "max_tokens": self._settings.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": [*_search_results(sources), {"type": "text", "text": question}]}
            ],
            "output_config": {"effort": self._settings.effort},
        }
        if self._settings.server_side_fallback:
            request["betas"] = [_FALLBACK_BETA]
            request["fallbacks"] = "default"

        try:
            async with self._client.beta.messages.stream(**request) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield AnswerText(text=event.text)
                    elif event.type == "citation" and event.citation.type == "search_result_location":
                        index = event.citation.search_result_index
                        if not 0 <= index < len(sources):
                            log.warning("generation.citation_out_of_range", index=index)
                            continue
                        yield AnswerCitation(
                            source_index=index,
                            chunk_id=str(sources[index].chunk.chunk_id),
                            cited_text=event.citation.cited_text,
                        )
                final = await stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise GenerationError(
                f"Claude API returned {exc.status_code} (request_id={exc.request_id})"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise GenerationError("could not reach the Claude API") from exc

        fell_back = any(block.type == "fallback" for block in final.content)
        log.info(
            "generation.complete",
            requested_model=self._settings.model,
            served_model=final.model,
            fell_back=fell_back,
            stop_reason=final.stop_reason,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            sources=len(sources),
            request_id=final._request_id,
        )
        yield AnswerComplete(
            stop_reason=final.stop_reason,
            model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            refused=final.stop_reason == "refusal",
        )


def _search_results(sources: Sequence[ScoredChunk]) -> list[BetaSearchResultBlockParam]:
    blocks: list[BetaSearchResultBlockParam] = []
    for s in sources:
        section = s.chunk.metadata.get("section")
        title = f"{s.chunk.title} — {section}" if section else s.chunk.title
        blocks.append(
            {
                "type": "search_result",
                "source": s.chunk.source_uri,
                "title": title,
                "content": [{"type": "text", "text": s.chunk.text}],
                "citations": {"enabled": True},
            }
        )
    return blocks
