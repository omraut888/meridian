"""Internal domain types.

These are frozen, slotted dataclasses rather than pydantic models: they are
constructed in hot loops over thousands of chunks, and validation already
happens at the system edges (config loading and the HTTP API).
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Fixed namespace so chunk IDs are stable across processes and machines.
CHUNK_NAMESPACE = uuid.UUID("6f1d3c52-8a0e-4c9b-9a55-3e2f1b7d4c10")

FloatArray = NDArray[np.float32]


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Document:
    """A source document prior to chunking.

    Attributes:
        doc_id: Stable, source-derived identifier (e.g. ``"arxiv:2401.01234"``).
        title: Human-readable title.
        text: Full plain-text body.
        source_uri: Canonical URL or path of the original.
        metadata: Flat string metadata propagated to every chunk.
    """

    doc_id: str
    title: str
    text: str
    source_uri: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the title and body; changes whenever indexed content would."""
        return content_hash(f"{self.title}\n{self.text}")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous span of a document, sized for embedding.

    Attributes:
        chunk_id: Deterministic UUIDv5 of ``(doc_id, content_hash, index)``.
        doc_id: Owning document.
        index: Zero-based position within the document.
        text: Chunk body as embedded and shown to the generator.
        token_count: Length in embedding-model tokens (sum over packed units).
        char_span: ``[start, end)`` offsets into the document text.
        title: Owning document's title.
        source_uri: Owning document's URI.
        content_hash: Owning document's content hash at ingestion time.
        metadata: Document metadata plus chunk-level fields (e.g. section).
    """

    chunk_id: uuid.UUID
    doc_id: str
    index: int
    text: str
    token_count: int
    char_span: tuple[int, int]
    title: str
    source_uri: str
    content_hash: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, doc_hash: str, index: int) -> uuid.UUID:
        """Derive the deterministic chunk ID used as the Qdrant point ID."""
        return uuid.uuid5(CHUNK_NAMESPACE, f"{doc_id}:{doc_hash}:{index}")


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A sparse term-weight vector (BM25 term frequencies; IDF applied server-side)."""

    indices: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A chunk with its dense and sparse representations."""

    chunk: Chunk
    dense: FloatArray
    sparse: SparseVector
    cluster_id: int | None = None


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieval candidate returned by the vector store.

    Attributes:
        chunk: The retrieved chunk.
        score: Stage score (RRF score after fusion; cosine for single-branch search).
        dense: The chunk's dense vector, needed for MMR.
        cluster_id: The chunk's cluster assignment, if clustering has run.
    """

    chunk: Chunk
    score: float
    dense: FloatArray
    cluster_id: int | None


@dataclass(frozen=True, slots=True)
class ClusterMatch:
    """A cluster centroid matched against a query."""

    cluster_id: int
    similarity: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Output of the retrieval pipeline for one query."""

    query: str
    chunks: list[ScoredChunk]
    routed_clusters: list[ClusterMatch]
    timings_ms: dict[str, float]
