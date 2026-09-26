"""Dense and sparse text embedders.

Both the write path (ingestion) and the read path (retrieval) embed text, so
embedders live in their own module rather than inside either pipeline.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Literal, Protocol

import numpy as np
import structlog
import voyageai.error
from fastembed import SparseTextEmbedding
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# voyageai re-exports these without __all__, which strict mypy treats as private.
from voyageai.client import Client
from voyageai.client_async import AsyncClient

from meridian.config import SparseSettings, VoyageSettings
from meridian.exceptions import ConfigurationError, EmbeddingError
from meridian.models import FloatArray, SparseVector

log = structlog.get_logger(__name__)

# Only failures that can succeed on retry. Auth and request-validation errors
# are raised immediately: retrying them wastes quota and hides the real bug.
# Voyage timestamps requests on arrival, and upload latency varies with batch
# size, so a request sent exactly 60 s after an earlier one can land inside that
# one's minute. A 60 s window produced one 429 per minute at the window edge.
_PACING_WINDOW_S = 62.0

_TRANSIENT_VOYAGE_ERRORS: tuple[type[Exception], ...] = (
    voyageai.error.RateLimitError,
    voyageai.error.ServiceUnavailableError,
    voyageai.error.ServerError,
    voyageai.error.Timeout,
    voyageai.error.APIConnectionError,
    voyageai.error.TryAgain,
)


TokenCounter = Callable[[Sequence[str]], list[int]]


def voyage_token_counter(model: str) -> TokenCounter:
    """Return a per-text token counter using ``model``'s tokenizer.

    Tokenization runs locally (the Hugging Face tokenizer is downloaded once and
    cached), needs no API key, and makes no network call per invocation.
    """
    client = Client(api_key=None)

    def count(texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        return [len(enc.ids) for enc in client.tokenize(list(texts), model=model)]

    return count


class RateLimiter:
    """Sliding-window limit on requests and tokens per window, with even spacing.

    ``acquire`` waits until a request of the given size fits both budgets, then
    records it. Requests are also spaced at least ``window / requests`` apart:
    providers often enforce "N per minute" as a smoothly refilling bucket, which
    rejects a burst of N even when the window has room. Pacing before sending,
    instead of retrying on 429s, matters under very low limits: rejected
    requests count against the request budget too, so retry storms can starve
    every batch.
    """

    def __init__(
        self,
        *,
        requests: int | None,
        tokens: int | None,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create the limiter.

        Args:
            requests: Maximum requests per window, or ``None`` for no request limit.
            tokens: Maximum tokens per window, or ``None`` for no token limit.
            window_s: Window length in seconds.
            clock: Monotonic clock (injectable for tests).
            sleep: Async sleep (injectable for tests).
        """
        self._requests = requests
        self._tokens = tokens
        self._window = window_s
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        """Wait until a request costing ``tokens`` fits, then record it.

        Raises:
            ValueError: If ``tokens`` exceeds the per-window token budget, so the
                request could never be sent.
        """
        if self._tokens is not None and tokens > self._tokens:
            raise ValueError(f"request of {tokens} tokens exceeds the {self._tokens}-token window budget")
        async with self._lock:  # FIFO: waiters are admitted in arrival order
            while True:
                now = self._clock()
                while self._sent and self._sent[0][0] <= now - self._window:
                    self._sent.popleft()
                wait = self._wait_for(now, tokens)
                if wait <= 0:
                    self._sent.append((now, tokens))
                    return
                await self._sleep(wait)

    def _wait_for(self, now: float, tokens: int) -> float:
        """Seconds until enough of the window expires for this request to fit."""
        wait = 0.0
        if self._requests is not None and self._sent:
            wait = self._sent[-1][0] + self._window / self._requests - now  # even spacing
            if len(self._sent) >= self._requests:
                wait = max(wait, self._sent[len(self._sent) - self._requests][0] + self._window - now)
        if self._tokens is not None:
            used = sum(t for _, t in self._sent)
            for sent_at, sent_tokens in self._sent:
                if used + tokens <= self._tokens:
                    break
                used -= sent_tokens
                wait = max(wait, sent_at + self._window - now)
        return wait


class DenseEmbedder(Protocol):
    """Asymmetric dense embedder (documents and queries are embedded differently)."""

    @property
    def dimension(self) -> int:
        """Output vector dimensionality."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        """Embed corpus passages; returns an ``(n, d)`` L2-normalized matrix."""
        ...

    async def embed_queries(self, texts: Sequence[str]) -> FloatArray:
        """Embed search queries; returns an ``(n, d)`` L2-normalized matrix."""
        ...

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Return per-text token counts under the embedding model's tokenizer."""
        ...


class SparseEmbedder(Protocol):
    """Lexical sparse embedder."""

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed corpus passages."""
        ...

    async def embed_queries(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed search queries."""
        ...


class VoyageEmbedder:
    """Dense embeddings from the Voyage AI API.

    Requests are packed by both text count and token budget so no batch exceeds
    provider limits, run with bounded concurrency, and retried with jittered
    exponential backoff on transient failures only.
    """

    def __init__(self, settings: VoyageSettings) -> None:
        """Create the embedder.

        Args:
            settings: Voyage configuration.

        Raises:
            ConfigurationError: If no API key is configured.
        """
        if settings.api_key is None:
            raise ConfigurationError("MERIDIAN_VOYAGE__API_KEY is not set")
        self._settings = settings
        # Retries are owned by this class (with logging), so the SDK's are disabled.
        self._client = AsyncClient(
            api_key=settings.api_key.get_secret_value(),
            max_retries=0,
            timeout=settings.timeout_s,
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._count_tokens = voyage_token_counter(settings.model)
        self._limiter = (
            RateLimiter(
                requests=settings.requests_per_minute,
                tokens=settings.tokens_per_minute,
                window_s=_PACING_WINDOW_S,
            )
            if settings.requests_per_minute or settings.tokens_per_minute
            else None
        )
        # A batch larger than the per-minute token budget could never be sent.
        self._batch_tokens = min(
            settings.max_batch_tokens, settings.tokens_per_minute or settings.max_batch_tokens
        )

    @property
    def dimension(self) -> int:
        """Output vector dimensionality."""
        return self._settings.output_dimension

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Return per-text token counts under the Voyage model's tokenizer."""
        return self._count_tokens(texts)

    async def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        """Embed corpus passages with ``input_type="document"``."""
        return await self._embed(texts, input_type="document")

    async def embed_queries(self, texts: Sequence[str]) -> FloatArray:
        """Embed search queries with ``input_type="query"``."""
        return await self._embed(texts, input_type="query")

    async def _embed(self, texts: Sequence[str], *, input_type: Literal["document", "query"]) -> FloatArray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches = list(self._pack_batches(texts))
        results = await asyncio.gather(
            *(self._embed_batch(batch, tokens, input_type=input_type) for batch, tokens in batches)
        )
        matrix = np.vstack(results).astype(np.float32, copy=False)
        return _l2_normalize(matrix)

    def _pack_batches(self, texts: Sequence[str]) -> Iterator[tuple[list[str], int]]:
        """Yield ``(batch, token_count)`` pairs within the text and token limits."""
        token_counts = self.count_tokens(texts)
        batch: list[str] = []
        batch_tokens = 0
        for text, n_tokens in zip(texts, token_counts, strict=True):
            full = len(batch) >= self._settings.max_batch_texts
            over_budget = batch_tokens + n_tokens > self._batch_tokens
            if batch and (full or over_budget):
                yield batch, batch_tokens
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n_tokens
        if batch:
            yield batch, batch_tokens

    async def _embed_batch(
        self, batch: list[str], tokens: int, *, input_type: Literal["document", "query"]
    ) -> FloatArray:
        if self._settings.tokens_per_minute is not None and tokens > self._settings.tokens_per_minute:
            # Only a single oversized text can get here; packing keeps batches within budget.
            raise EmbeddingError(
                f"text of {tokens} tokens exceeds tokens_per_minute={self._settings.tokens_per_minute}"
            )
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_TRANSIENT_VOYAGE_ERRORS),
            wait=wait_random_exponential(multiplier=1.0, max=60.0),
            stop=stop_after_attempt(self._settings.max_retries + 1),
            before_sleep=_log_retry,
            reraise=True,
        )
        try:
            async with self._semaphore:
                async for attempt in retrying:
                    with attempt:
                        if self._limiter is not None:  # retries are paced too
                            await self._limiter.acquire(tokens)
                        response = await self._client.embed(
                            batch,
                            model=self._settings.model,
                            input_type=input_type,
                            output_dimension=self._settings.output_dimension,
                            truncation=False,
                        )
        except voyageai.error.VoyageError as exc:
            raise EmbeddingError(f"Voyage embed failed for batch of {len(batch)} ({input_type})") from exc
        log.debug(
            "voyage.embed",
            batch_size=len(batch),
            input_type=input_type,
            total_tokens=response.total_tokens,
        )
        return np.asarray(response.embeddings, dtype=np.float32)


class BM25Embedder:
    """BM25 sparse vectors via FastEmbed.

    FastEmbed emits the term-frequency half of BM25; Qdrant applies IDF at query
    time (``Modifier.IDF``), so corpus statistics stay correct as documents are
    added without re-embedding. Inference is CPU-bound and runs off the event loop.
    """

    def __init__(self, settings: SparseSettings) -> None:
        """Load the sparse model (downloads weights on first use)."""
        self._model = SparseTextEmbedding(model_name=settings.model)

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed corpus passages."""
        return await asyncio.to_thread(self._embed_documents_sync, list(texts))

    async def embed_queries(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed search queries (query-side BM25 weighting)."""
        return await asyncio.to_thread(self._embed_queries_sync, list(texts))

    def _embed_documents_sync(self, texts: list[str]) -> list[SparseVector]:
        return [_to_sparse(e.indices, e.values) for e in self._model.embed(texts)]

    def _embed_queries_sync(self, texts: list[str]) -> list[SparseVector]:
        return [_to_sparse(e.indices, e.values) for e in self._model.query_embed(texts)]


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "voyage.embed.retry",
        attempt=state.attempt_number,
        sleep_s=round(state.next_action.sleep, 2) if state.next_action else None,
        error=type(exc).__name__ if exc else None,
    )


def _to_sparse(indices: np.ndarray, values: np.ndarray) -> SparseVector:
    return SparseVector(indices=tuple(int(i) for i in indices), values=tuple(float(v) for v in values))


def _l2_normalize(matrix: FloatArray) -> FloatArray:
    norms: FloatArray = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingError("provider returned a zero vector")
    # numpy's stubs widen float32 / float32 to floating[Any]; this is a no-op cast.
    return (matrix / norms).astype(np.float32, copy=False)
