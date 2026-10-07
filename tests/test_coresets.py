import numpy as np
import pytest

from tabpfn_lab import coresets as cs
from tabpfn_lab.backends import SklearnBackend, align_proba, softmax
from tabpfn_lab.sizes import make_synthetic

METHODS = sorted(cs.CORESET_METHODS)
KW = {"uncertainty_mix": {"pilot": 100}, "embedding_kmeans": {"pilot": 100}}


@pytest.fixture(scope="module")
def data():
    X, y = make_synthetic(300, 5, n_classes=3, seed=1)
    return X, y


@pytest.fixture(scope="module")
def reg_data():
    return make_synthetic(300, 5, task="regression", seed=2)


def sel(method, X, y, b, seed=0, task="classification"):
    return cs.select_coreset(method, X, y, b, seed, task, **KW.get(method, {}))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_size_and_uniqueness(method, task, data, reg_data):
    X, y = data if task == "classification" else reg_data
    for b in (5, 40, 150):
        idx = sel(method, X, y, b, task=task)
        assert len(idx) == min(b, len(X))
        assert len(np.unique(idx)) == len(idx)
        assert idx.min() >= 0 and idx.max() < len(X)
        assert np.issubdtype(idx.dtype, np.integer)


@pytest.mark.parametrize("method", METHODS)
def test_deterministic_and_differs_across_seeds(method, data):
    X, y = data
    a, b = sel(method, X, y, 40, seed=3), sel(method, X, y, 40, seed=3)
    np.testing.assert_array_equal(a, b)
    c = sel(method, X, y, 40, seed=4)
    assert set(a.tolist()) != set(c.tolist())


@pytest.mark.parametrize("method", METHODS)
def test_budget_ge_n_returns_all(method, data):
    X, y = data
    for b in (len(X), len(X) + 10):
        idx = sel(method, X, y, b)
        np.testing.assert_array_equal(np.sort(idx), np.arange(len(X)))


@pytest.mark.parametrize("method", METHODS)
def test_budget_zero_or_negative_raises(method, data):
    X, y = data
    for b in (0, -3):
        with pytest.raises(ValueError):
            sel(method, X, y, b)


@pytest.mark.parametrize("method", sorted(set(METHODS) - cs.CLASS_COVERAGE_EXEMPT))
def test_class_coverage(method):
    X, y = make_synthetic(300, 5, imbalance=[0.9, 0.07, 0.03], noise=0.0, seed=3)
    for b in (3, 5, 20):
        for seed in range(3):
            idx = cs.select_coreset(method, X, y, b, seed, **KW.get(method, {}))
            assert set(np.unique(y[idx])) == set(np.unique(y)), (method, b, seed)


def test_stratified_proportions(data):
    X, y = data
    full = np.bincount(y) / len(y)
    for b in (30, 90):
        idx = cs.select_coreset("stratified", X, y, b, 0)
        frac = np.bincount(y[idx], minlength=3) / b
        assert np.all(np.abs(frac - full) <= 1 / b + 0.02)


def test_balanced_is_balanced_and_flagged(data):
    X, y = data
    idx = cs.select_coreset("balanced", X, y, 60, 0)
    counts = np.bincount(y[idx])
    assert counts.max() - counts.min() <= 1
    assert cs.method_info("balanced")["prior_shifting"]
    assert not cs.method_info("stratified")["prior_shifting"]


def test_kmeans_picks_real_points(data):
    X, y = data
    idx = cs.select_coreset("kmeans_stratified", X, y, 30, 0)
    assert np.issubdtype(idx.dtype, np.integer) and len(np.unique(idx)) == 30
    for i in idx:  # every returned index is a real row; no synthetic centroid sneaks in
        assert 0 <= i < len(X)


def test_uncertainty_mix_fraction():
    # two well-separated blobs plus an overlapping strip: the strip is obviously uncertain
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(0, 1, (n, 2))
    y = (X[:, 0] > 0).astype(int)
    X[:, 0] *= 3
    b = 60
    idx = cs.select_coreset("uncertainty_mix", X, y, b, seed=1, frac_hard=0.5, pilot=200)
    u = cs.uncertainty_scores(X, y, seed=1, task="classification", pilot=200)
    threshold = np.sort(u)[::-1][b // 2 - 1]  # tie-inclusive top-uncertainty set (forest entropies tie a lot)
    top = set(np.flatnonzero(u >= threshold).tolist())
    hits = len(top & set(idx.tolist()))
    assert hits >= 0.45 * b


def test_no_label_leak_to_test(data):
    X, y = data
    n_train = 200
    X_train, y_train = X[:n_train], y[:n_train]
    for method in METHODS:
        idx = cs.select_coreset(method, X_train, y_train, 50, 0, **KW.get(method, {}))
        assert idx.max() < n_train  # never points into the appended test rows


def test_perest_independent(data):
    X, y = data
    subs = cs.per_estimator_subsets("perest_random", X, y, 50, 4, 0)
    assert len(subs) == 4 and all(len(s) == 50 for s in subs)
    assert len({tuple(sorted(s.tolist())) for s in subs}) == 4


@pytest.mark.parametrize("E,b", [(4, 75), (4, 100), (8, 40), (3, 300)])
def test_perest_partition_cover(data, E, b):
    X, y = data
    n = len(X)
    subs = cs.per_estimator_subsets("perest_partition", X, y, b, E, 0)
    assert all(len(s) == min(b, n) and len(np.unique(s)) == len(s) for s in subs)
    if E * b >= n:
        assert set(np.concatenate(subs).tolist()) == set(range(n))


def test_handles_missing_class_downstream(data):
    X, y = data
    keep = np.flatnonzero(y != 2)[:60]
    be = SklearnBackend(n_trees=5)
    out = be.clf_outputs(X[keep], y[keep], X[:20], np.arange(3), 3, 0)
    assert out.logits.shape == (3, 20, 3)
    P = softmax(out.logits).mean(0)
    np.testing.assert_allclose(P.sum(1), 1)
    assert P[:, 2].max() < 1e-12
    p = align_proba(np.array([[0.3, 0.7]]), [0, 1], [0, 1, 2])
    np.testing.assert_allclose(p, [[0.3, 0.7, 0.0]])


def test_prior_correct_restores_prior():
    p = np.array([[0.5, 0.5], [0.8, 0.2]])
    out = cs.prior_correct(p, context_prior=[0.5, 0.5], target_prior=[0.9, 0.1])
    np.testing.assert_allclose(out[0], [0.9, 0.1])
    np.testing.assert_allclose(out.sum(1), 1)
