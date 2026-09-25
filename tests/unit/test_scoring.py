import uuid

import numpy as np
import pytest

from meridian.evaluation import bootstrap_ci, score_ranking
from meridian.models import Chunk, ScoredChunk


def _chunk(doc_id: str, vector: list[float]) -> ScoredChunk:
    v = np.asarray(vector, dtype=np.float32)
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=uuid.uuid4(),
            doc_id=doc_id,
            index=0,
            text="",
            token_count=0,
            char_span=(0, 0),
            title="",
            source_uri="",
            content_hash="",
        ),
        score=0.0,
        dense=v / np.linalg.norm(v),
        cluster_id=None,
    )


def test_relevant_doc_counts_once_at_its_first_chunk() -> None:
    ranking = [_chunk("a", [1, 0]), _chunk("b", [0, 1]), _chunk("b", [0, 1]), _chunk("c", [1, 1])]
    scores = score_ranking(ranking, frozenset({"b"}), k=10, latency_ms=1.0)

    assert scores.first_relevant_rank == 2
    assert scores.ndcg_at_k == pytest.approx(1 / np.log2(3))
    assert scores.mrr_at_k == pytest.approx(0.5)
    assert scores.recall_at_5 == 1.0
    assert scores.distinct_docs == 3


def test_miss_beyond_cutoff_scores_zero() -> None:
    ranking = [_chunk("a", [1, 0]), _chunk("b", [0, 1])]
    scores = score_ranking(ranking, frozenset({"b"}), k=1, latency_ms=0.0)

    assert scores.first_relevant_rank is None
    assert scores.ndcg_at_k == 0.0
    assert scores.recall_at_k == 0.0


def test_multiple_relevant_docs_normalize_by_ideal_dcg() -> None:
    ranking = [_chunk("x", [1, 0]), _chunk("a", [0, 1]), _chunk("b", [1, 1])]
    scores = score_ranking(ranking, frozenset({"a", "b"}), k=3, latency_ms=0.0)

    ideal = 1 + 1 / np.log2(3)
    assert scores.ndcg_at_k == pytest.approx((1 / np.log2(3) + 1 / np.log2(4)) / ideal)
    assert scores.recall_at_k == 1.0


def test_intra_list_similarity_of_identical_vectors_is_one() -> None:
    ranking = [_chunk("a", [1, 0]), _chunk("b", [1, 0]), _chunk("c", [1, 0])]
    assert score_ranking(
        ranking, frozenset({"a"}), k=3, latency_ms=0.0
    ).intra_list_similarity == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    values = np.random.default_rng(1).random(200)
    low, high = bootstrap_ci(values)

    assert low < values.mean() < high
    assert bootstrap_ci(values) == (low, high)
