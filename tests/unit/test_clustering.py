import numpy as np
import pytest

from meridian.clustering import CorpusClusterer
from meridian.config import ClusteringSettings


def _blobs(n_per: int, centers: int, dim: int = 32, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Tight clusters around orthogonal directions on the unit sphere."""
    rng = np.random.default_rng(seed)
    basis = np.eye(dim, dtype=np.float32)[:centers]
    x = np.vstack([basis[i] + noise * rng.standard_normal((n_per, dim)) for i in range(centers)])
    labels = np.repeat(np.arange(centers), n_per)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), labels


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    mapping: dict[int, int] = {}
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        if mapping.setdefault(x, y) != y:
            return False
    return len(set(mapping.values())) == len(mapping)


def test_silhouette_selects_true_k_and_recovers_partition() -> None:
    x, truth = _blobs(n_per=40, centers=6)
    model = CorpusClusterer(ClusteringSettings(k_grid=(4, 6, 8, 10))).fit(x)

    assert model.k == 6
    assert max(model.silhouette_by_k, key=model.silhouette_by_k.__getitem__) == 6
    assert _same_partition(model.labels, truth)


def test_centroids_are_unit_norm_and_labels_match_nearest_centroid() -> None:
    x, _ = _blobs(n_per=25, centers=4)
    model = CorpusClusterer(ClusteringSettings(n_clusters=4)).fit(x)

    np.testing.assert_allclose(np.linalg.norm(model.centroids, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(model.labels, np.argmax(x @ model.centroids.T, axis=1))


def test_fixed_k_is_honored_and_version_is_deterministic() -> None:
    x, _ = _blobs(n_per=20, centers=3)
    settings = ClusteringSettings(n_clusters=5)
    a, b = CorpusClusterer(settings).fit(x), CorpusClusterer(settings).fit(x)

    assert a.k == 5
    assert a.version == b.version


def test_rejects_too_few_vectors() -> None:
    x, _ = _blobs(n_per=1, centers=3)
    with pytest.raises(ValueError, match="cannot cluster"):
        CorpusClusterer(ClusteringSettings(n_clusters=3)).fit(x)
