import logging
import time
import tracemalloc

import numpy as np
import pytest

from tabpfn_lab import coresets as cs
from tabpfn_lab.cache import OutputCache
from tabpfn_lab.sizes import TIERS, make_synthetic, resolve_budgets, tier_of

METHODS = sorted(cs.CORESET_METHODS)
SIZES = [300, 3_000, 30_000]
RATIOS = [0.5, 0.1, 0.01]
KW = {"uncertainty_mix": {"pilot": 300}, "embedding_kmeans": {"pilot": 300}}
_DATA: dict = {}


def data(n, n_features=6, n_classes=3, **kw):
    key = (n, n_features, n_classes, tuple(sorted(kw.items())))
    if key not in _DATA:
        _DATA[key] = make_synthetic(n, n_features, n_classes=n_classes, seed=n % 97, **kw)
    return _DATA[key]


def sel(method, X, y, b, seed=0, task="classification"):
    return cs.select_coreset(method, X, y, b, seed, task, **KW.get(method, {}))


def check_contract(idx, n, b, y=None, method=None):
    assert len(idx) == min(b, n)
    assert len(np.unique(idx)) == len(idx)
    assert idx.min() >= 0 and idx.max() < n
    assert np.issubdtype(idx.dtype, np.integer)
    if y is not None and method not in cs.CLASS_COVERAGE_EXEMPT and b >= len(np.unique(y)):
        assert set(np.unique(y[idx])) == set(np.unique(y))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("ratio", RATIOS)
def test_contract_across_sizes_and_ratios(method, n, ratio):
    X, y = data(n)
    b = max(int(round(ratio * n)), 3)
    idx = sel(method, X, y, b)
    check_contract(idx, n, b, y, method)
    np.testing.assert_array_equal(idx, sel(method, X, y, b))


@pytest.mark.parametrize("method", METHODS)
def test_contract_edge_budgets(method):
    n = 300
    X, y = data(n)
    for b in (n - 1, n, n + 1, 3, 1):
        idx = sel(method, X, y, b)
        check_contract(idx, n, b, y, method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", SIZES)
def test_extreme_reduction(method, n):
    X, y = data(n)
    b = resolve_budgets(n, fractions=[0.001], n_classes=3)[0]  # clipped up to a sane minimum
    check_contract(sel(method, X, y, b), n, b, y, method)


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_selection_time_scales(method):
    def t(n):
        X, y = data(n)
        best = np.inf
        for _ in range(2):
            t0 = time.perf_counter()
            sel(method, X, y, int(0.1 * n))
            best = min(best, time.perf_counter() - t0)
        return best

    small, big = t(3_000), t(30_000)
    assert big < 30 * max(small, 0.01), (method, small, big)


@pytest.mark.parametrize("method", METHODS)
def test_selection_memory_bounded(method):
    X, y = data(30_000, n_features=20)
    tracemalloc.start()
    sel(method, X, y, 3_000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 10 * X.nbytes, (method, peak / 2**20, X.nbytes / 2**20)


@pytest.mark.parametrize("n", SIZES)
def test_imbalanced_classes(n):
    X, y = data(n, imbalance=0.01, noise=0.0)
    minority = 2
    misses = 0
    for seed in range(5):
        for method in ("stratified", "kmeans_stratified"):
            idx = cs.select_coreset(method, X, y, 10, seed)
            assert (y[idx] == minority).any(), (method, n, seed)
        misses += not (y[cs.select_coreset("random", X, y, 10, seed)] == minority).any()
    # random is allowed to miss; just record how often
    print(f"n={n}: random missed the minority class in {misses}/5 draws")


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("n_classes", [10, 50])
def test_many_classes(n, n_classes):
    X, y = data(n, n_classes=n_classes, noise=0.0)
    b = max(int(0.1 * n), n_classes)
    for method in ("stratified", "balanced", "kmeans_stratified", "uncertainty_mix"):
        idx = sel(method, X, y, b)
        assert len(np.unique(y[idx])) == len(np.unique(y)), method


@pytest.mark.parametrize("d", [5, 50, 500])
def test_wide_tables(d):
    X, y = data(3_000, n_features=d)
    for method in ("kmeans_stratified", "embedding_kmeans"):
        idx = sel(method, X, y, 200)
        check_contract(idx, 3_000, 200, y, method)
    assert cs.features(X).shape[1] <= cs.MAX_DIMS


@pytest.mark.parametrize("method", METHODS)
def test_messy_inputs(method):
    rng = np.random.default_rng(0)
    X, y = make_synthetic(600, 6, n_classes=3, seed=4)
    X = X.copy()
    X[rng.random(X.shape) < 0.1] = np.nan
    X[:, 1] = 7.0  # constant column
    X[:, 2] = rng.integers(0, 5, len(X))  # categorical codes
    X[100:150] = X[:50]  # duplicate rows
    y[100:150] = y[:50]
    a, b = sel(method, X, y, 60), sel(method, X, y, 60)
    check_contract(a, 600, 60, y, method)
    np.testing.assert_array_equal(a, b)
    Xr, yr = make_synthetic(600, 6, task="regression", heavy_tail=True, seed=5)
    Xr = Xr.copy()
    Xr[rng.random(Xr.shape) < 0.1] = np.nan
    r1, r2 = sel(method, Xr, yr, 60, task="regression"), sel(method, Xr, yr, 60, task="regression")
    check_contract(r1, 600, 60)
    np.testing.assert_array_equal(r1, r2)


def test_regression_tail_coverage():
    X, y = make_synthetic(3_000, 5, task="regression", heavy_tail=True, seed=6)
    b = 300
    top = set(np.argsort(-y)[: len(y) // 100].tolist())
    share = len(top) * b / len(y)  # proportional share of the top 1%
    for seed in range(5):
        idx = cs.select_coreset("stratified", X, y, b, seed, "regression")
        assert len(top & set(idx.tolist())) >= 0.5 * share


def test_resolve_budgets_rules(caplog):
    with caplog.at_level(logging.INFO, logger="tabpfn_lab"):
        out = resolve_budgets(1000, fractions=[0.5, 0.1, 0.001, 0.1], absolute=[500, 2000, 1], n_classes=5)
    assert out == sorted(set(out))
    assert out == [5, 100, 500]  # 0.001*1000=1 and absolute 1 clipped to 5; 2000 >= n_train dropped
    assert any("clipped" in r.message for r in caplog.records)
    assert any("dropped" in r.message for r in caplog.records)
    assert resolve_budgets(1000, [0.5, 0.1], [500], 5) == resolve_budgets(1000, [0.5, 0.1], [500], 5)
    assert min(resolve_budgets(50, [0.01], None, None)) >= 2


def test_tier_assignment():
    assert tier_of(999) == "tiny" and tier_of(1000) == "small"
    assert tier_of(9_999) == "small" and tier_of(10_000) == "medium"
    assert tier_of(99_999) == "medium" and tier_of(100_000) == "large"
    assert tier_of(50) == "tiny" and tier_of(5_000_000) == "large"
    assert set(TIERS) == {"tiny", "small", "medium", "large"}


def test_runner_handles_mixed_tiers(smoke):
    from tabpfn_lab import stats
    from tabpfn_lab.cli import validate_csv
    from tabpfn_lab.experiments import read_results, run_coreset

    from conftest import SpyBackend

    ds = [
        {"name": "t_tiny", "source": "synthetic", "task": "classification", "n": 600, "imbalance": 0.2, "data_seed": 1},
        {"name": "t_small", "source": "synthetic", "task": "classification", "n": 2_000, "imbalance": 0.2, "data_seed": 2},
        {"name": "t_medium", "source": "synthetic", "task": "classification", "n": 14_500, "imbalance": 0.2, "data_seed": 3},
    ]
    cfg = smoke(
        datasets={"dev": [], "confirm": ds},
        coreset_methods=["random", "stratified", "builtin_subsample", "builtin_majority_downsample"],
        budget_fractions=[0.3],
        selection_seeds=[0, 1],
        n_estimators=1,
        backend={"name": "sklearn", "n_trees": 5},
    )
    csv = run_coreset(cfg, backend=SpyBackend(n_trees=5))
    df = read_results(csv)
    assert dict(df.groupby("dataset")["tier"].first()) == {"t_tiny": "tiny", "t_small": "small", "t_medium": "medium"}
    assert validate_csv(csv) == []
    res = stats.compare_by_tier(df[df["budget_key"] == "f0.3"], "stratified", "random", "log_loss")
    assert set(res["tiers"]) == {"tiny", "small", "medium"}
    assert all(r["n_datasets"] == 1 for r in res["tiers"].values())
    assert res["pooled"]["n_datasets"] == 3


def test_cache_keys_include_size_factors():
    base = dict(dataset="d", task="classification", split_seed=0, n_train=1000, n_test=300, backend={"name": "x"}, n_estimators=8, subset_id="random:100:{}")
    k = OutputCache.make_key(**base)
    assert OutputCache.make_key(**{**base, "subset_id": "random:200:{}"}) != k  # budget
    assert OutputCache.make_key(**{**base, "subset_id": "stratified:100:{}"}) != k  # method
    assert OutputCache.make_key(**{**base, "n_train": 100_000}) != k
    assert OutputCache.make_key(**{**base, "overrides": {"SUBSAMPLE_SAMPLES": 100}}) != k


@pytest.mark.tabpfn
@pytest.mark.tier_large
def test_tier_large_smoke(tmp_path):
    from tabpfn_lab.experiments import read_results, run_coreset

    from conftest import load_smoke

    cfg = load_smoke(
        tmp_path,
        backend={"name": "tabpfn", "version": "3.5", "device": "cuda"},
        datasets={"dev": [], "confirm": [{"name": "covertype", "source": "openml", "data_id": 1596, "task": "classification", "max_rows": 160_000}]},
        coreset_methods=["random"],
        budget_fractions=[0.1, 0.01],
        selection_seeds=[0],
        n_estimators=4,
        learning_curve_points=0,
    )
    csv = run_coreset(cfg)
    df = read_results(csv)
    fails = csv.with_name(csv.stem + ".failures.csv")
    assert df["n_train"].iloc[0] >= 100_000
    assert {"full", "random"} <= set(df["method"]) | ({"full"} if fails.exists() else set())
