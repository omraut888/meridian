"""Dense and sparse text embedders.

Both the write path (ingestion) and the read path (retrieval) embed text, so
embedders live in their own module rather than inside either pipeline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
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
            *(self._embed_batch(batch, input_type=input_type) for batch in batches)
        )
        matrix = np.vstack(results).astype(np.float32, copy=False)
        return _l2_normalize(matrix)

    def _pack_batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        token_counts = self.count_tokens(texts)
        batch: list[str] = []
        batch_tokens = 0
        for text, n_tokens in zip(texts, token_counts, strict=True):
            full = len(batch) >= self._settings.max_batch_texts
            over_budget = batch_tokens + n_tokens > self._settings.max_batch_tokens
            if batch and (full or over_budget):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n_tokens
        if batch:
            yield batch

    async def _embed_batch(self, batch: list[str], *, input_type: Literal["document", "query"]) -> FloatArray:
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
