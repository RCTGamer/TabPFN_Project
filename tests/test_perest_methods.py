import numpy as np
import pytest

from tabpfn_lab import aggregation as A
from tabpfn_lab import cli, experiments
from tabpfn_lab import coresets as cs
from tabpfn_lab.experiments import (
    local_ratio,
    read_results,
    read_timing,
    run_coreset,
    timing_path,
)
from tabpfn_lab.sizes import make_synthetic

NEW = ["perest_stratified", "perest_kcenter", "cluster_experts"]


@pytest.fixture
def data():
    return make_synthetic(600, 5, n_classes=3, imbalance=[0.6, 0.3, 0.1], seed=0)


@pytest.mark.parametrize("method", NEW)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_new_perest_contract(method, task):
    X, y = make_synthetic(500, 4, n_classes=3, task=task, seed=1)
    a = cs.per_estimator_subsets(method, X, y, 60, 4, 3, task)
    b = cs.per_estimator_subsets(method, X, y, 60, 4, 3, task)
    assert len(a) == 4
    for s, t in zip(a, b):
        assert len(s) == 60 and len(np.unique(s)) == 60 and s.min() >= 0 and s.max() < len(X)
        np.testing.assert_array_equal(s, t)
        if task == "classification":
            assert set(np.unique(y[s])) == set(np.unique(y))
    assert len({tuple(sorted(s.tolist())) for s in a}) > 1


def test_feature_transform_maps_new_rows_like_training_rows(data):
    X, _ = data
    T = cs.feature_transform(X[:400], seed=2)
    np.testing.assert_array_equal(T(X[:400]), cs.features(X[:400], seed=2))
    assert T(X[400:]).shape == (200, X.shape[1])


def test_kcenter_covers_better_than_random():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(2000, 2))
    y = np.zeros(2000, int)
    F = cs.features(X)

    def radius(idx):
        return np.sqrt(((F[:, None, :] - F[idx][None]) ** 2).sum(-1).min(axis=1).max())

    kc = cs.per_estimator_subsets("perest_kcenter", X, y, 50, 1, 0)[0]
    rnd = cs.per_estimator_subsets("perest_random", X, y, 50, 1, 0)[0]
    assert radius(kc) < 0.7 * radius(rnd)


def test_cluster_experts_specialise_and_gate_is_a_distribution(data):
    X, y = data
    subs = cs.per_estimator_subsets("cluster_experts", X, y, 40, 4, 0)
    labels = cs._expert_clusters(X, 4, 0)[2]
    for e, s in enumerate(subs):  # mostly rows of the estimator's own cluster
        assert np.mean(labels[s] == e) > 0.9
    w = cs.cluster_gate(X[:500], X[500:], 4, 0)
    assert w.shape == (4, 100) and (w >= 0).all()
    np.testing.assert_allclose(w.sum(axis=0), 1.0)


def test_sample_weighted_uniform_equals_mean(logits_factory):
    L = logits_factory(E=4, n=30, C=3)
    w = np.full((4, 30), 0.25)
    np.testing.assert_allclose(A.agg_sample_weighted(L, 1.3, w=w), A.agg_mean(L, 1.3), atol=1e-12)
    P = A.agg_sample_weighted(L, 1.0, w=np.random.default_rng(0).random((4, 30)))
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        A.agg_sample_weighted(L, 1.0, w=np.ones(4))


@pytest.mark.parametrize("ratio", [0.5, 1.0])
def test_local_mix_routing(data, ratio):
    X, y = data
    Xtr, ytr, Xte = X[:450], y[:450], X[450:]
    groups = cs.local_mix_routing(Xtr, ytr, Xte, 60, 3, 0, "classification", ratio, n_groups=4)
    rows = np.concatenate([r for r, _ in groups])
    assert sorted(rows.tolist()) == list(range(len(Xte)))  # every test row in exactly one group
    T = cs.feature_transform(Xtr, 0)
    Ftr = T(Xtr)
    for r, subsets in groups:
        centre = T(Xte[r]).mean(axis=0)
        nearest = set(np.argsort(((Ftr - centre) ** 2).sum(axis=1))[:240].tolist())
        assert len(subsets) == 3 and len({tuple(sorted(s.tolist())) for s in subsets}) == 3
        for s in subsets:
            assert len(s) == 60 and len(np.unique(s)) == 60 and set(np.unique(ytr[s])) == {0, 1, 2}
            near = np.mean([i in nearest for i in s])
            assert near > (0.9 if ratio == 1.0 else 0.45)
            if ratio == 0.5:
                assert near < 0.85


def _cfg(tmp_path, methods, **kw):
    cfg = {
        "name": "perest_t",
        "backend": {"name": "sklearn", "n_trees": 5},
        "datasets": {
            "dev": [
                {"name": "clf", "source": "synthetic", "task": "classification", "n": 600, "n_features": 5, "n_classes": 3, "imbalance": [0.6, 0.3, 0.1], "noise": 0.02, "data_seed": 1},
                {"name": "reg", "source": "synthetic", "task": "regression", "n": 500, "n_features": 4, "data_seed": 2},
            ],
            "confirm": [],
        },
        "n_estimators": 4,
        "budget_fractions": [],
        "budget_absolute": [60],
        "coreset_methods": methods,
        "local_ratios": [0.5, 1.0],
        "local_groups": 3,
        "baseline": "perest_random",
        "split_seeds": 1,
        "selection_seeds": 2,
        "cache_dir": None,
        "log_dir": str(tmp_path / "logs"),
        "out_csv": str(tmp_path / "t" / "perest_t.csv"),
    }
    cfg.update(kw)
    return cfg


METHODS = ["perest_random", "perest_partition", "builtin_subsample", "perest_stratified", "perest_representative", "perest_kcenter", "local_mix", "cluster_experts"]


def test_perest_run_flags_timing_analysis_and_determinism(tmp_path):
    csv = run_coreset(_cfg(tmp_path / "a", METHODS))
    assert cli.validate_csv(csv) == []
    df = read_results(csv)
    have = set(df["method"])
    assert {"local_mix_r0.5", "local_mix_r1", "cluster_experts+gate", "full"} <= have
    loc = df["method"].map(local_ratio).notna()
    assert (df.loc[loc, "flags"] == "transductive").all() and (df.loc[~loc, "flags"] == "").all()
    t = read_timing(timing_path(csv))
    assert set(t["cell"]) == set(df["cell"])
    assert (t.loc[t["method"] == "local_mix_r0.5", "n_fits"] == 3).all()  # one fit per test group
    assert (t.loc[t["method"] == "perest_random", "n_fits"] == 1).all()
    assert (t[["fit_seconds", "predict_seconds", "selection_seconds"]] >= 0).all().all()
    dev = cli.analyze(csv)["dev"]
    assert dev.loc[dev["method"] == "perest_random", "improvement"].isna().all()
    assert dev.loc[dev["method"] == "local_mix_r1", "improvement"].notna().all()
    assert dev["total_seconds"].notna().all()
    again = run_coreset(_cfg(tmp_path / "b", METHODS))
    assert csv.read_bytes() == again.read_bytes()


def test_local_selection_never_sees_test_labels(tmp_path, monkeypatch):
    seen = []
    real = experiments.cs.local_mix_routing

    def spy(X_train, y_train, X_test, *a, **kw):
        seen.append((len(X_train), len(y_train), len(X_test)))
        return real(X_train, y_train, X_test, *a, **kw)

    monkeypatch.setattr(experiments.cs, "local_mix_routing", spy)
    cfg = _cfg(tmp_path, ["perest_random", "local_mix"], local_ratios=[0.9], datasets={"dev": [_cfg(tmp_path, [])["datasets"]["dev"][0]], "confirm": []})
    run_coreset(cfg)
    split = experiments.Context(experiments.normalize_config(cfg, "coreset"), None).split("clf", 0)
    assert seen and all(s == (split.n_train, split.n_train, split.n_test) for s in seen)  # labels only for training rows


def test_local_mix_expands_from_ratios():
    cfg = experiments.normalize_config({"datasets": [], "coreset_methods": ["random", "local_mix"], "local_ratios": [0.9, 1]}, "coreset")
    assert cfg["coreset_methods"] == ["random", "local_mix_r0.9", "local_mix_r1"]
    assert experiments.normalize_config(cfg, "coreset")["coreset_methods"] == cfg["coreset_methods"]
    assert local_ratio("local_mix_r0.75+median") == 0.75 and local_ratio("perest_random") is None
