"""Hybrid retrieval: dense + sparse recall, RRF fusion, cluster routing, and MMR."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import structlog
from numpy.typing import NDArray

from meridian.clustering import ClusterRouter
from meridian.config import RetrievalSettings
from meridian.embeddings import DenseEmbedder, SparseEmbedder
from meridian.exceptions import ClusterModelMissingError
from meridian.models import ClusterMatch, FloatArray, RetrievalResult, ScoredChunk, SparseVector
from meridian.vector_store import QdrantVectorStore

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    """Per-query pipeline configuration.

    Every stage can be switched off independently, which is what lets the
    evaluation harness ablate them against the same index.

    Attributes:
        use_dense: Include the dense (Voyage) branch.
        use_sparse: Include the sparse (BM25) branch.
        route_top_m: Restrict the dense branch to the ``m`` nearest clusters;
            ``0`` disables routing. The sparse branch is never restricted.
        mmr_lambda: MMR relevance/diversity trade-off in ``[0, 1]``; ``None``
            disables MMR and returns the fused ranking as-is.
        top_k: Number of chunks returned.
        candidate_pool: Fused candidates handed to MMR.
        prefetch_limit: Per-branch depth before fusion.
    """

    use_dense: bool = True
    use_sparse: bool = True
    route_top_m: int = 0
    mmr_lambda: float | None = 0.7
    top_k: int = 8
    candidate_pool: int = 40
    prefetch_limit: int = 100

    def __post_init__(self) -> None:
        if not (self.use_dense or self.use_sparse):
            raise ValueError("at least one retrieval branch must be enabled")
        if self.mmr_lambda is not None and not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if self.top_k < 1 or self.candidate_pool < self.top_k:
            raise ValueError("require 1 <= top_k <= candidate_pool")

    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> RetrievalOptions:
        """Build the default options from configuration."""
        return cls(
            route_top_m=settings.route_top_m,
            mmr_lambda=settings.mmr_lambda,
            top_k=settings.top_k,
            candidate_pool=settings.candidate_pool,
            prefetch_limit=settings.prefetch_limit,
        )

    def with_overrides(self, **changes: object) -> RetrievalOptions:
        """Return a copy with the given non-``None`` fields replaced."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    """Precomputed dense and sparse representations of a query."""

    dense: FloatArray
    sparse: SparseVector


def mmr(relevance: NDArray[np.floating], candidates: FloatArray, *, k: int, lambda_: float) -> list[int]:
    """Select ``k`` items by Maximal Marginal Relevance.

    Greedily picks the item maximizing
    ``lambda_ * relevance[i] - (1 - lambda_) * max_{j in selected} sim(i, j)``,
    maintaining the running max-similarity vector incrementally: ``O(k * n * d)``.

    Args:
        relevance: ``(n,)`` relevance scores, ideally on a ``[0, 1]`` scale.
        candidates: ``(n, d)`` unit-norm candidate vectors.
        k: Number of items to select (clipped to ``n``).
        lambda_: ``1.0`` is pure relevance; ``0.0`` is pure diversity.

    Returns:
        Indices into ``candidates`` in selection order.
    """
    n = candidates.shape[0]
    if relevance.shape != (n,):
        raise ValueError(f"relevance shape {relevance.shape} does not match {n} candidates")
    k = min(k, n)
    if k <= 0:
        return []

    rel = relevance.astype(np.float64)
    max_sim = np.full(n, -np.inf)
    available = np.ones(n, dtype=bool)
    selected: list[int] = []
    for _ in range(k):
        scores = rel if not selected else lambda_ * rel - (1.0 - lambda_) * max_sim
        scores = np.where(available, scores, -np.inf)
        best = int(np.argmax(scores))
        selected.append(best)
        available[best] = False
        max_sim = np.maximum(max_sim, candidates @ candidates[best])
    return selected


def _minmax(scores: NDArray[np.floating]) -> NDArray[np.float64]:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores, dtype=np.float64)
    return (scores.astype(np.float64) - lo) / (hi - lo)


class HybridRetriever:
    """Query-time retrieval pipeline.

    Stages: embed (dense and sparse concurrently) → route the query to its
    nearest clusters → one Qdrant hybrid query (dense branch restricted to the
    routed clusters, sparse branch global, RRF-fused server-side) → MMR.

    MMR's relevance term is the min-max-normalized *fused* score rather than
    raw query cosine, so re-ranking diversifies the hybrid ranking instead of
    silently collapsing it back to a dense-only ordering.
    """

    def __init__(
        self,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        store: QdrantVectorStore,
        router: ClusterRouter,
        defaults: RetrievalOptions,
    ) -> None:
        """Create the retriever."""
        self._dense = dense
        self._sparse = sparse
        self._store = store
        self._router = router
        self.defaults = defaults

    async def embed_queries(self, queries: Sequence[str]) -> list[QueryEmbedding]:
        """Embed a batch of queries in one dense and one sparse call."""
        dense, sparse = await asyncio.gather(
            self._dense.embed_queries(queries), self._sparse.embed_queries(queries)
        )
        return [QueryEmbedding(dense=d, sparse=s) for d, s in zip(dense, sparse, strict=True)]

    async def retrieve(self, query: str, options: RetrievalOptions | None = None) -> RetrievalResult:
        """Embed ``query`` and run the full pipeline.

        Args:
            query: Natural-language query.
            options: Per-query options; defaults to the configured pipeline.

        Returns:
            Ranked chunks, routed clusters, and per-stage timings.
        """
        started = time.perf_counter()
        (embedding,) = await self.embed_queries([query])
        embed_ms = (time.perf_counter() - started) * 1000
        result = await self.retrieve_embedded(query, embedding, options)
        result.timings_ms["embed"] = round(embed_ms, 2)
        result.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    async def retrieve_embedded(
        self, query: str, embedding: QueryEmbedding, options: RetrievalOptions | None = None
    ) -> RetrievalResult:
        """Run the pipeline for an already-embedded query.

        Args:
            query: The query text (for logging and the result).
            embedding: Precomputed query vectors.
            options: Per-query options; defaults to the configured pipeline.

        Returns:
            Ranked chunks, routed clusters, and per-stage timings.
        """
        opts = options or self.defaults
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        routed: list[ClusterMatch] = []
        if opts.use_dense and opts.route_top_m > 0:
            try:
                routed = await self._router.route(embedding.dense, opts.route_top_m)
            except ClusterModelMissingError:
                log.warning("retrieval.routing_skipped", reason="no cluster model")
        timings["route"] = _ms_since(t0)

        t0 = time.perf_counter()
        use_mmr = opts.mmr_lambda is not None
        candidates = await self._store.search(
            dense=embedding.dense if opts.use_dense else None,
            sparse=embedding.sparse if opts.use_sparse else None,
            limit=opts.candidate_pool if use_mmr else opts.top_k,
            prefetch_limit=opts.prefetch_limit,
            dense_cluster_filter=[m.cluster_id for m in routed] or None,
        )
        timings["search"] = _ms_since(t0)

        t0 = time.perf_counter()
        if use_mmr and len(candidates) > opts.top_k:
            assert opts.mmr_lambda is not None
            order = mmr(
                _minmax(np.array([c.score for c in candidates])),
                np.vstack([c.dense for c in candidates]),
                k=opts.top_k,
                lambda_=opts.mmr_lambda,
            )
            chunks: list[ScoredChunk] = [candidates[i] for i in order]
        else:
            chunks = candidates[: opts.top_k]
        timings["rerank"] = _ms_since(t0)

        log.info(
            "retrieval.complete",
            query_chars=len(query),
            candidates=len(candidates),
            returned=len(chunks),
            routed_clusters=[m.cluster_id for m in routed],
            **{f"{k}_ms": v for k, v in timings.items()},
        )
        return RetrievalResult(query=query, chunks=chunks, routed_clusters=routed, timings_ms=timings)


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
