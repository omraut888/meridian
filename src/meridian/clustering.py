"""Corpus clustering and query-to-cluster routing.

Chunks are clustered with spherical k-means: k-means on L2-normalized vectors
with re-normalized centroids, which approximates clustering under cosine
similarity, the same metric Qdrant searches with. k-means was chosen over
density-based methods (DBSCAN/HDBSCAN) because every chunk must belong to a
cluster to be routable; density methods label sparse regions as noise.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from meridian.config import ClusteringSettings
from meridian.exceptions import ClusterModelMissingError
from meridian.models import ClusterMatch, FloatArray
from meridian.vector_store import QdrantVectorStore

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ClusterModel:
    """A fitted clustering of the corpus.

    Attributes:
        centroids: ``(k, d)`` unit-norm centroid matrix.
        labels: Cluster ID per input vector, aligned with the fit input.
        k: Number of clusters.
        silhouette: Cosine silhouette score of the fit (sampled for large corpora).
        version: Content-derived identifier of the centroids.
        silhouette_by_k: Scores for every candidate k evaluated during selection.
    """

    centroids: FloatArray
    labels: NDArray[np.int32]
    k: int
    silhouette: float
    version: str
    silhouette_by_k: dict[int, float]


class CorpusClusterer:
    """Fit spherical k-means, choosing k by silhouette when not fixed.

    ``fit`` is pure and synchronous (no I/O) so it is trivially testable and can
    be offloaded to a worker thread by async callers.
    """

    def __init__(self, settings: ClusteringSettings) -> None:
        """Create the clusterer."""
        self._settings = settings

    def fit(self, vectors: FloatArray) -> ClusterModel:
        """Cluster ``vectors``.

        Args:
            vectors: ``(n, d)`` embedding matrix; normalized internally.

        Returns:
            The fitted model.

        Raises:
            ValueError: If there are too few vectors for any candidate k.
        """
        x = _normalize(np.asarray(vectors, dtype=np.float32))
        n = x.shape[0]
        candidates = (
            [self._settings.n_clusters]
            if self._settings.n_clusters is not None
            else [k for k in self._settings.k_grid if 2 <= k < n]
        )
        if not candidates or candidates[0] >= n:
            raise ValueError(f"cannot cluster {n} vectors with k in {candidates}")

        scores: dict[int, float] = {}
        best: tuple[float, int, MiniBatchKMeans] | None = None
        for k in candidates:
            model = MiniBatchKMeans(
                n_clusters=k,
                n_init=5,
                batch_size=1024,
                random_state=self._settings.random_state,
            ).fit(x)
            score = self._silhouette(x, model.labels_)
            scores[k] = round(score, 4)
            log.debug("clustering.candidate", k=k, silhouette=score)
            if best is None or score > best[0]:
                best = (score, k, model)

        assert best is not None  # candidates is non-empty
        score, k, model = best
        centroids = _normalize(model.cluster_centers_.astype(np.float32))
        # Assign with the normalized centroids so labels match query-time routing.
        labels = np.argmax(x @ centroids.T, axis=1).astype(np.int32)
        version = f"k{k}-{hashlib.sha256(centroids.tobytes()).hexdigest()[:12]}"
        log.info("clustering.fit", n=n, k=k, silhouette=round(score, 4), version=version)
        return ClusterModel(
            centroids=centroids,
            labels=labels,
            k=k,
            silhouette=score,
            version=version,
            silhouette_by_k=scores,
        )

    def _silhouette(self, x: FloatArray, labels: NDArray[np.int32]) -> float:
        if len(np.unique(labels)) < 2:
            return -1.0
        sample = min(self._settings.silhouette_sample_size, x.shape[0])
        return float(
            silhouette_score(
                x, labels, metric="cosine", sample_size=sample, random_state=self._settings.random_state
            )
        )


async def recluster_corpus(store: QdrantVectorStore, clusterer: CorpusClusterer) -> ClusterModel:
    """Refit clusters over every indexed chunk and persist the result.

    Args:
        store: Source of vectors and destination of the model.
        clusterer: The clustering algorithm.

    Returns:
        The fitted model.

    Raises:
        ClusterModelMissingError: If the collection is empty.
    """
    ids: list[uuid.UUID] = []
    batches: list[FloatArray] = []
    async for batch_ids, batch_vectors in store.iter_dense_vectors():
        ids.extend(batch_ids)
        batches.append(batch_vectors)
    if not ids:
        raise ClusterModelMissingError("no chunks indexed; run ingestion first")

    model = await asyncio.to_thread(clusterer.fit, np.vstack(batches))
    assignments = dict(zip(ids, (int(label) for label in model.labels), strict=True))
    await store.write_cluster_model(model.centroids, assignments, model.version)
    return model


class ClusterRouter:
    """Route queries (and newly ingested chunks) to clusters."""

    def __init__(self, store: QdrantVectorStore) -> None:
        """Create the router."""
        self._store = store
        self._centroids: FloatArray | None = None
        self._loaded = False

    async def route(self, query: FloatArray, top_m: int) -> list[ClusterMatch]:
        """Return the ``top_m`` clusters nearest to ``query``.

        Raises:
            ClusterModelMissingError: If no cluster model has been fitted.
        """
        matches = await self._store.search_centroids(query, top_m)
        if not matches:
            raise ClusterModelMissingError("no centroids stored; run `meridian recluster`")
        return matches

    async def assign(self, vectors: FloatArray) -> Sequence[int | None]:
        """Assign each vector to its nearest centroid, or ``None`` if unclustered."""
        if not self._loaded:
            await self.refresh()
        if self._centroids is None:
            return [None] * vectors.shape[0]
        return [int(i) for i in np.argmax(vectors @ self._centroids.T, axis=1)]

    async def refresh(self) -> None:
        """Reload centroids from the store (call after a recluster)."""
        self._centroids = await self._store.load_centroids()
        self._loaded = True


def _normalize(x: FloatArray) -> FloatArray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)
