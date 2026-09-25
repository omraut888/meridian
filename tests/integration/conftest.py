"""Live fixtures: a real Qdrant, Voyage AI, and Claude, on throwaway collections.

Skipped unless both API keys are configured and Qdrant is reachable. Run with
``pytest -m integration``; the default run deselects these tests.
"""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from meridian.clustering import recluster_corpus
from meridian.config import Settings
from meridian.ingestion import IngestionReport, load_documents
from meridian.models import Document
from meridian.services import Services

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus" / "documents.jsonl"

# Three articles per topic plus one arXiv abstract each: small enough to embed
# for a few cents, broad enough that clustering and routing have real structure.
SLICE = frozenset(
    {
        "wikipedia:40226710",  # Raft (algorithm)
        "wikipedia:970031",  # Byzantine fault
        "wikipedia:26411268",  # CAP theorem
        "wikipedia:25385",  # RSA cryptosystem
        "wikipedia:1260",  # Advanced Encryption Standard
        "wikipedia:7903",  # Diffie-Hellman key exchange
        "wikipedia:61603971",  # Transformer (deep learning)
        "wikipedia:43561218",  # Word embedding
        "wikipedia:1906608",  # Named-entity recognition
        "wikipedia:1208345",  # Scale-invariant feature transform
        "wikipedia:15822591",  # Object detection
        "wikipedia:40409788",  # Convolutional neural network
        "wikipedia:4674",  # B-tree
        "wikipedia:38562148",  # Log-structured merge-tree
        "wikipedia:244953",  # Write-ahead logging
        "wikipedia:373371",  # Static single-assignment form
        "wikipedia:485122",  # Register allocation
        "wikipedia:18030",  # LR parser
        "arxiv:2501.00337",
        "arxiv:2501.00517",
        "arxiv:2501.00656",
        "arxiv:2501.00654",
        "arxiv:2412.20871",
        "arxiv:2501.00169",
    }
)
N_CLUSTERS = 6


@dataclass(frozen=True, slots=True)
class IndexedCorpus:
    services: Services
    documents: tuple[Document, ...]
    first_run: IngestionReport


def _live_settings() -> Settings:
    base = Settings()
    if base.voyage.api_key is None or base.generation.api_key is None:
        pytest.skip("MERIDIAN_VOYAGE__API_KEY and MERIDIAN_GENERATION__API_KEY must be set")
    run = uuid.uuid4().hex[:8]
    return base.model_copy(
        update={
            "qdrant": base.qdrant.model_copy(
                update={
                    "chunks_collection": f"meridian_it_{run}_chunks",
                    "centroids_collection": f"meridian_it_{run}_centroids",
                }
            ),
            "clustering": base.clustering.model_copy(update={"n_clusters": N_CLUSTERS}),
        }
    )


@pytest.fixture(scope="session")
async def indexed() -> AsyncIterator[IndexedCorpus]:
    settings = _live_settings()
    services = await Services.create(settings)
    try:
        try:
            await services.store.ping()
        except Exception as exc:  # noqa: BLE001 - any failure means "no Qdrant", so skip
            pytest.skip(f"Qdrant not reachable at {settings.qdrant.url}: {exc}")
        documents = tuple(d for d in load_documents(CORPUS) if d.doc_id in SLICE)
        assert len(documents) == len(SLICE), "slice references documents missing from the corpus"

        report = await services.ingestion_pipeline().run(documents)
        await recluster_corpus(services.store, services.clusterer())
        await services.router.refresh()
        yield IndexedCorpus(services, documents, report)
    finally:
        await services.store.drop_collections()
        await services.aclose()
