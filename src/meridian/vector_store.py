"""Qdrant persistence for chunks and cluster centroids.

A single chunks collection holds two named vectors per point: ``dense``
(Voyage, cosine) and ``sparse`` (BM25 TF with server-side IDF). Hybrid search
is one Query API call with two prefetch branches fused by Reciprocal Rank
Fusion inside Qdrant, so there is no client-side fusion and one network round
trip per query.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import structlog
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from meridian.config import QdrantSettings
from meridian.exceptions import SchemaMismatchError, VectorStoreError
from meridian.models import (
    Chunk,
    ClusterMatch,
    EmbeddedChunk,
    FloatArray,
    ScoredChunk,
    SparseVector,
)

log = structlog.get_logger(__name__)

DENSE = "dense"
SPARSE = "sparse"


@contextmanager
def _qdrant_errors(operation: str) -> Iterator[None]:
    """Translate Qdrant client failures into :class:`VectorStoreError`."""
    try:
        yield
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        raise VectorStoreError(f"Qdrant {operation} failed: {exc}") from exc


class QdrantVectorStore:
    """Async repository over the chunks and centroids collections."""

    def __init__(self, client: AsyncQdrantClient, settings: QdrantSettings, dense_dim: int) -> None:
        """Create the store.

        Args:
            client: A shared async Qdrant client (owned by the caller).
            settings: Collection names and batching configuration.
            dense_dim: Dimensionality of the dense vectors.
        """
        self._client = client
        self._settings = settings
        self._dense_dim = dense_dim
        self._chunks = settings.chunks_collection
        self._centroids = settings.centroids_collection

    @classmethod
    def connect(cls, settings: QdrantSettings, dense_dim: int) -> QdrantVectorStore:
        """Build a store with its own client from settings."""
        client = AsyncQdrantClient(
            url=settings.url,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            timeout=settings.timeout_s,
        )
        return cls(client, settings, dense_dim)

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.close()

    # ------------------------------------------------------------------ schema

    async def ensure_schema(self) -> None:
        """Create collections and payload indexes if absent; validate them if present.

        Raises:
            SchemaMismatchError: If an existing collection was built with a
                different dimensionality, distance, or sparse modifier.
        """
        with _qdrant_errors("ensure_schema"):
            if await self._client.collection_exists(self._chunks):
                await self._validate_chunks_schema()
            else:
                await self._client.create_collection(
                    self._chunks,
                    vectors_config={
                        DENSE: models.VectorParams(size=self._dense_dim, distance=models.Distance.COSINE)
                    },
                    sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
                )
                log.info("qdrant.collection.created", collection=self._chunks)
            # Idempotent: Qdrant no-ops when the index already exists.
            await self._client.create_payload_index(self._chunks, "doc_id", models.PayloadSchemaType.KEYWORD)
            await self._client.create_payload_index(
                self._chunks, "cluster_id", models.PayloadSchemaType.INTEGER
            )

            if not await self._client.collection_exists(self._centroids):
                await self._client.create_collection(
                    self._centroids,
                    vectors_config=models.VectorParams(size=self._dense_dim, distance=models.Distance.COSINE),
                )
                log.info("qdrant.collection.created", collection=self._centroids)

    async def _validate_chunks_schema(self) -> None:
        info = await self._client.get_collection(self._chunks)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        dense = vectors.get(DENSE) if isinstance(vectors, dict) else None
        problems: list[str] = []
        if dense is None:
            problems.append(f"missing named vector {DENSE!r}")
        else:
            if dense.size != self._dense_dim:
                problems.append(f"dense size {dense.size} != configured {self._dense_dim}")
            if dense.distance != models.Distance.COSINE:
                problems.append(f"dense distance {dense.distance} != Cosine")
        sparse_params = sparse.get(SPARSE)
        if sparse_params is None:
            problems.append(f"missing sparse vector {SPARSE!r}")
        elif sparse_params.modifier != models.Modifier.IDF:
            problems.append("sparse vector lacks IDF modifier")
        if problems:
            raise SchemaMismatchError(
                f"collection {self._chunks!r} is incompatible: {'; '.join(problems)}. "
                "Drop it or point MERIDIAN_QDRANT__CHUNKS_COLLECTION elsewhere."
            )

    async def drop_collections(self) -> None:
        """Delete both collections (used by tests and full rebuilds)."""
        with _qdrant_errors("drop_collections"):
            for name in (self._chunks, self._centroids):
                if await self._client.collection_exists(name):
                    await self._client.delete_collection(name)

    # ------------------------------------------------------------------ writes

    async def get_content_hash(self, doc_id: str) -> str | None:
        """Return the stored content hash for ``doc_id``, or ``None`` if not indexed."""
        with _qdrant_errors("get_content_hash"):
            points, _ = await self._client.scroll(
                self._chunks,
                scroll_filter=_match("doc_id", doc_id),
                limit=1,
                with_payload=["content_hash"],
                with_vectors=False,
            )
        if not points or points[0].payload is None:
            return None
        value = points[0].payload.get("content_hash")
        return str(value) if value is not None else None

    async def replace_document(self, doc_id: str, chunks: Sequence[EmbeddedChunk]) -> None:
        """Atomically-enough swap a document's chunks for a new version.

        New chunks are upserted first, then any chunk of the same document with a
        different content hash is deleted. Chunk IDs embed the content hash, so
        old and new versions never collide, and a crash between the two steps
        leaves duplicates (self-healing on the next run) rather than a gap.

        Args:
            doc_id: Document being replaced.
            chunks: All chunks of the new version; must share one content hash.
        """
        if not chunks:
            return
        hashes = {c.chunk.content_hash for c in chunks}
        if len(hashes) != 1 or any(c.chunk.doc_id != doc_id for c in chunks):
            raise ValueError("chunks must all belong to one document version")
        new_hash = hashes.pop()

        with _qdrant_errors("replace_document"):
            size = self._settings.upsert_batch_size
            for start in range(0, len(chunks), size):
                await self._client.upsert(
                    self._chunks,
                    points=[_to_point(c) for c in chunks[start : start + size]],
                    wait=True,
                )
            await self._client.delete(
                self._chunks,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))],
                        must_not=[
                            models.FieldCondition(key="content_hash", match=models.MatchValue(value=new_hash))
                        ],
                    )
                ),
                wait=True,
            )

    async def count_chunks(self) -> int:
        """Return the exact number of indexed chunks."""
        with _qdrant_errors("count"):
            result = await self._client.count(self._chunks, exact=True)
        return result.count

    # ------------------------------------------------------------------ search

    async def search(
        self,
        *,
        dense: FloatArray | None,
        sparse: SparseVector | None,
        limit: int,
        prefetch_limit: int,
        dense_cluster_filter: Sequence[int] | None = None,
    ) -> list[ScoredChunk]:
        """Run dense, sparse, or hybrid (RRF-fused) search.

        Args:
            dense: Query dense vector, or ``None`` to skip the dense branch.
            sparse: Query sparse vector, or ``None`` to skip the sparse branch.
            limit: Number of candidates to return.
            prefetch_limit: Per-branch candidate depth before fusion.
            dense_cluster_filter: If given, restrict the *dense* branch to these
                cluster IDs. The sparse branch always searches the full corpus.

        Returns:
            Candidates in descending score order, with dense vectors attached.
        """
        if dense is None and sparse is None:
            raise ValueError("at least one of dense or sparse must be provided")

        dense_filter = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="cluster_id", match=models.MatchAny(any=list(dense_cluster_filter))
                    )
                ]
            )
            if dense_cluster_filter
            else None
        )
        dense_query = dense.tolist() if dense is not None else None
        sparse_query = (
            models.SparseVector(indices=list(sparse.indices), values=list(sparse.values))
            if sparse is not None
            else None
        )

        common: dict[str, Any] = {
            "collection_name": self._chunks,
            "limit": limit,
            "with_payload": True,
            "with_vectors": [DENSE],
        }
        with _qdrant_errors("search"):
            if dense_query is not None and sparse_query is not None:
                response = await self._client.query_points(
                    **common,
                    prefetch=[
                        models.Prefetch(
                            query=dense_query, using=DENSE, limit=prefetch_limit, filter=dense_filter
                        ),
                        models.Prefetch(query=sparse_query, using=SPARSE, limit=prefetch_limit),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                )
            elif dense_query is not None:
                response = await self._client.query_points(
                    **common, query=dense_query, using=DENSE, query_filter=dense_filter
                )
            else:
                response = await self._client.query_points(**common, query=sparse_query, using=SPARSE)
        return [_to_scored(p) for p in response.points]

    # ---------------------------------------------------------------- clusters

    async def iter_dense_vectors(
        self, batch_size: int = 1024
    ) -> AsyncIterator[tuple[list[uuid.UUID], FloatArray]]:
        """Yield ``(ids, vectors)`` batches covering every chunk in the collection."""
        offset: models.ExtendedPointId | None = None
        while True:
            with _qdrant_errors("scroll"):
                points, offset = await self._client.scroll(
                    self._chunks,
                    limit=batch_size,
                    offset=offset,
                    with_payload=False,
                    with_vectors=[DENSE],
                )
            if points:
                ids = [uuid.UUID(str(p.id)) for p in points]
                vectors = np.asarray([_dense_of(p) for p in points], dtype=np.float32)
                yield ids, vectors
            if offset is None:
                return

    async def iter_payloads(
        self, fields: Sequence[str], batch_size: int = 1024
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield the selected payload fields of every chunk."""
        offset: models.ExtendedPointId | None = None
        while True:
            with _qdrant_errors("scroll"):
                points, offset = await self._client.scroll(
                    self._chunks,
                    limit=batch_size,
                    offset=offset,
                    with_payload=list(fields),
                    with_vectors=False,
                )
            for point in points:
                yield point.payload or {}
            if offset is None:
                return

    async def write_cluster_model(
        self, centroids: FloatArray, assignments: Mapping[uuid.UUID, int], version: str
    ) -> None:
        """Persist centroids and per-chunk cluster assignments.

        Args:
            centroids: ``(k, d)`` unit-norm centroid matrix; row ``i`` is cluster ``i``.
            assignments: Chunk ID to cluster ID.
            version: Identifier of the fitted model, stored on every centroid.
        """
        by_cluster: dict[int, list[str]] = defaultdict(list)
        for point_id, cluster_id in assignments.items():
            by_cluster[cluster_id].append(str(point_id))

        with _qdrant_errors("write_cluster_model"):
            # Replace centroids wholesale: k may have changed between fits.
            await self._client.delete(
                self._centroids,
                points_selector=models.FilterSelector(filter=models.Filter()),
                wait=True,
            )
            await self._client.upsert(
                self._centroids,
                points=[
                    models.PointStruct(
                        id=i,
                        vector=centroids[i].tolist(),
                        payload={"version": version, "size": len(by_cluster.get(i, []))},
                    )
                    for i in range(centroids.shape[0])
                ],
                wait=True,
            )
            size = self._settings.upsert_batch_size * 8
            for cluster_id, ids in by_cluster.items():
                for start in range(0, len(ids), size):
                    await self._client.set_payload(
                        self._chunks,
                        payload={"cluster_id": cluster_id},
                        points=ids[start : start + size],
                        wait=True,
                    )

    async def search_centroids(self, query: FloatArray, top_m: int) -> list[ClusterMatch]:
        """Return the ``top_m`` centroids most similar to ``query``."""
        with _qdrant_errors("search_centroids"):
            response = await self._client.query_points(
                self._centroids, query=query.tolist(), limit=top_m, with_payload=False
            )
        return [ClusterMatch(cluster_id=int(p.id), similarity=float(p.score)) for p in response.points]

    async def load_centroids(self) -> FloatArray | None:
        """Return the ``(k, d)`` centroid matrix ordered by cluster ID, or ``None``."""
        with _qdrant_errors("load_centroids"):
            points, _ = await self._client.scroll(
                self._centroids, limit=10_000, with_payload=False, with_vectors=True
            )
        if not points:
            return None
        points = sorted(points, key=lambda p: int(p.id))
        return np.asarray([p.vector for p in points], dtype=np.float32)

    # ------------------------------------------------------------------ health

    async def ping(self) -> None:
        """Raise :class:`VectorStoreError` if Qdrant is unreachable."""
        with _qdrant_errors("ping"):
            await self._client.get_collections()


def _match(key: str, value: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])


def _to_point(item: EmbeddedChunk) -> models.PointStruct:
    c = item.chunk
    payload: dict[str, Any] = {
        "doc_id": c.doc_id,
        "index": c.index,
        "text": c.text,
        "token_count": c.token_count,
        "char_start": c.char_span[0],
        "char_end": c.char_span[1],
        "title": c.title,
        "source_uri": c.source_uri,
        "content_hash": c.content_hash,
        "metadata": dict(c.metadata),
    }
    if item.cluster_id is not None:
        payload["cluster_id"] = item.cluster_id
    return models.PointStruct(
        id=str(c.chunk_id),
        vector={
            DENSE: item.dense.tolist(),
            SPARSE: models.SparseVector(indices=list(item.sparse.indices), values=list(item.sparse.values)),
        },
        payload=payload,
    )


def _dense_of(point: models.ScoredPoint | models.Record) -> list[float]:
    vector = point.vector
    if not isinstance(vector, dict) or DENSE not in vector:
        raise VectorStoreError(f"point {point.id} returned without a dense vector")
    dense = vector[DENSE]
    if not isinstance(dense, list):
        raise VectorStoreError(f"point {point.id} has a non-list dense vector")
    return dense  # type: ignore[return-value]


def _to_scored(point: models.ScoredPoint) -> ScoredChunk:
    payload = point.payload or {}
    try:
        chunk = Chunk(
            chunk_id=uuid.UUID(str(point.id)),
            doc_id=payload["doc_id"],
            index=int(payload["index"]),
            text=payload["text"],
            token_count=int(payload["token_count"]),
            char_span=(int(payload["char_start"]), int(payload["char_end"])),
            title=payload["title"],
            source_uri=payload["source_uri"],
            content_hash=payload["content_hash"],
            metadata=payload.get("metadata", {}),
        )
    except KeyError as exc:
        raise VectorStoreError(f"point {point.id} is missing payload field {exc}") from exc
    cluster_id = payload.get("cluster_id")
    return ScoredChunk(
        chunk=chunk,
        score=float(point.score),
        dense=np.asarray(_dense_of(point), dtype=np.float32),
        cluster_id=int(cluster_id) if cluster_id is not None else None,
    )
