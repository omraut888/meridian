import numpy as np
import pytest

from meridian.retrieval import RetrievalOptions, mmr


def _unit(*rows: list[float]) -> np.ndarray:
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_prefers_orthogonal_item_over_near_duplicate() -> None:
    # 0 and 1 are near-duplicates; 2 is orthogonal but slightly less relevant.
    candidates = _unit([1, 0, 0], [0.999, 0.045, 0], [0, 1, 0])
    relevance = np.array([1.0, 0.99, 0.9])

    assert mmr(relevance, candidates, k=2, lambda_=0.5) == [0, 2]


def test_lambda_one_is_pure_relevance_order() -> None:
    candidates = _unit([1, 0, 0], [0.999, 0.045, 0], [0, 1, 0], [0, 0, 1])
    relevance = np.array([0.2, 0.9, 0.5, 0.7])

    assert mmr(relevance, candidates, k=4, lambda_=1.0) == [1, 3, 2, 0]


def test_lambda_zero_first_pick_is_most_relevant_then_most_novel() -> None:
    candidates = _unit([1, 0, 0], [0.9, 0.1, 0], [0, 0, 1])
    relevance = np.array([1.0, 0.95, 0.1])

    assert mmr(relevance, candidates, k=2, lambda_=0.0) == [0, 2]


def test_k_is_clipped_and_indices_are_unique() -> None:
    candidates = _unit([1, 0], [0, 1], [1, 1])
    picked = mmr(np.array([0.3, 0.2, 0.1]), candidates, k=10, lambda_=0.7)

    assert sorted(picked) == [0, 1, 2]


def test_empty_selection() -> None:
    assert mmr(np.array([1.0]), _unit([1, 0]), k=0, lambda_=0.5) == []


def test_relevance_shape_must_match() -> None:
    with pytest.raises(ValueError, match="does not match"):
        mmr(np.array([1.0, 0.5]), _unit([1, 0]), k=1, lambda_=0.5)


def test_options_reject_no_branches_and_bad_pool() -> None:
    with pytest.raises(ValueError, match="branch"):
        RetrievalOptions(use_dense=False, use_sparse=False)
    with pytest.raises(ValueError, match="top_k"):
        RetrievalOptions(top_k=10, candidate_pool=5)


def test_with_overrides_ignores_none() -> None:
    base = RetrievalOptions(top_k=8, mmr_lambda=0.7)
    assert base.with_overrides(top_k=None, mmr_lambda=0.3) == RetrievalOptions(top_k=8, mmr_lambda=0.3)
