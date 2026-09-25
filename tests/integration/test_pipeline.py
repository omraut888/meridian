"""End-to-end pipeline tests against live Qdrant, Voyage AI, and Claude."""

import pytest

from meridian.generation import AnswerCitation, AnswerComplete, AnswerText
from meridian.retrieval import RetrievalOptions

from .conftest import N_CLUSTERS, IndexedCorpus

pytestmark = pytest.mark.integration

# Paraphrased descriptions (not titles) so retrieval has to match meaning, not a heading.
KNOWN_ITEMS = [
    (
        "consensus algorithm built on leader election and a replicated log, designed to be understandable",
        "wikipedia:40226710",
    ),
    (
        "public-key scheme whose security rests on how hard it is to factor a product of two large primes",
        "wikipedia:25385",
    ),
    (
        "neural network architecture based on multi-head self-attention instead of recurrence",
        "wikipedia:61603971",
    ),
    (
        "detecting keypoints with descriptors that are invariant to image scale and rotation",
        "wikipedia:1208345",
    ),
    ("self-balancing tree that keeps sorted keys in wide nodes for disk-based storage", "wikipedia:4674"),
    ("reaching agreement when some components may fail arbitrarily or act maliciously", "wikipedia:970031"),
]


async def test_first_ingest_indexes_every_document(indexed: IndexedCorpus) -> None:
    report = indexed.first_run
    assert report.failures == {}
    assert report.documents_seen == len(indexed.documents)
    assert report.documents_indexed == len(indexed.documents)
    assert report.chunks_upserted > 0
    assert await indexed.services.store.count_chunks() == report.chunks_upserted


async def test_reingest_skips_unchanged_documents(indexed: IndexedCorpus) -> None:
    before = await indexed.services.store.count_chunks()
    report = await indexed.services.ingestion_pipeline().run(indexed.documents)

    assert report.failures == {}
    assert report.documents_unchanged == len(indexed.documents)
    assert report.documents_indexed == 0
    assert report.chunks_upserted == 0
    assert await indexed.services.store.count_chunks() == before


@pytest.mark.parametrize(("query", "doc_id"), KNOWN_ITEMS)
async def test_known_item_is_retrieved(indexed: IndexedCorpus, query: str, doc_id: str) -> None:
    result = await indexed.services.retriever.retrieve(query)
    ranked = [c.chunk.doc_id for c in result.chunks]
    assert doc_id in ranked, ranked


async def test_routing_and_mmr_on_real_vectors(indexed: IndexedCorpus) -> None:
    query, _ = KNOWN_ITEMS[0]
    options = RetrievalOptions(route_top_m=2, mmr_lambda=0.5, top_k=6, candidate_pool=24)
    result = await indexed.services.retriever.retrieve(query, options)

    assert 1 <= len(result.routed_clusters) <= 2
    assert all(0 <= m.cluster_id < N_CLUSTERS for m in result.routed_clusters)
    assert len(result.chunks) == 6
    assert len({c.chunk.chunk_id for c in result.chunks}) == 6
    assert all(c.cluster_id is not None for c in result.chunks)
    assert all(c.dense.shape == (indexed.services.dense.dimension,) for c in result.chunks)


async def test_generated_answer_cites_retrieved_chunks(indexed: IndexedCorpus) -> None:
    question = "How does Raft elect a leader, and what happens when the leader fails?"
    sources = (await indexed.services.retriever.retrieve(question)).chunks
    events = [e async for e in indexed.services.generator.stream_answer(question, sources)]

    complete = events[-1]
    assert isinstance(complete, AnswerComplete)
    assert sum(isinstance(e, AnswerComplete) for e in events) == 1
    assert not complete.refused, complete
    assert complete.stop_reason == "end_turn", complete

    text = "".join(e.text for e in events if isinstance(e, AnswerText))
    citations = [e for e in events if isinstance(e, AnswerCitation)]
    assert text.strip()
    assert citations, "answer carried no citations"
    for citation in citations:
        source = sources[citation.source_index]
        assert citation.chunk_id == str(source.chunk.chunk_id)
        assert citation.cited_text.strip() in source.chunk.text
