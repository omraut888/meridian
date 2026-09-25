"""Composition root: builds and tears down the long-lived pipeline components.

Shared by the API (inside its lifespan) and the CLI, so both wire components
identically and there is exactly one place that knows the dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from meridian.clustering import ClusterRouter, CorpusClusterer
from meridian.config import Settings
from meridian.embeddings import BM25Embedder, VoyageEmbedder
from meridian.generation import ClaudeGenerator
from meridian.ingestion import IngestionPipeline, StructuralChunker
from meridian.retrieval import HybridRetriever, RetrievalOptions
from meridian.vector_store import QdrantVectorStore


@dataclass(slots=True)
class Services:
    """Wired pipeline components sharing one set of clients."""

    settings: Settings
    store: QdrantVectorStore
    dense: VoyageEmbedder
    sparse: BM25Embedder
    router: ClusterRouter
    retriever: HybridRetriever
    _generator: ClaudeGenerator | None = None

    @classmethod
    async def create(cls, settings: Settings) -> Services:
        """Build components and ensure the Qdrant schema exists.

        Raises:
            ConfigurationError: If required credentials are missing.
            SchemaMismatchError: If existing collections are incompatible.
        """
        dense = VoyageEmbedder(settings.voyage)
        sparse = BM25Embedder(settings.sparse)
        store = QdrantVectorStore.connect(settings.qdrant, dense_dim=dense.dimension)
        try:
            await store.ensure_schema()
        except BaseException:
            await store.close()
            raise
        router = ClusterRouter(store)
        retriever = HybridRetriever(
            dense, sparse, store, router, RetrievalOptions.from_settings(settings.retrieval)
        )
        return cls(settings, store, dense, sparse, router, retriever)

    @property
    def generator(self) -> ClaudeGenerator:
        """The answer generator, created on first use (retrieval-only use needs no key)."""
        if self._generator is None:
            self._generator = ClaudeGenerator(self.settings.generation)
        return self._generator

    def ingestion_pipeline(self) -> IngestionPipeline:
        """Build an ingestion pipeline over these components."""
        chunker = StructuralChunker(self.dense.count_tokens, self.settings.chunking)
        return IngestionPipeline(chunker, self.dense, self.sparse, self.store, self.router)

    def clusterer(self) -> CorpusClusterer:
        """Build a clusterer from configuration."""
        return CorpusClusterer(self.settings.clustering)

    async def aclose(self) -> None:
        """Release network clients."""
        await self.store.close()
        if self._generator is not None:
            await self._generator.close()
