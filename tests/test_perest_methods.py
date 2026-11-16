import numpy as np
import pytest

from tabpfn_lab import aggregation as A
from tabpfn_lab import cli, experiments
from tabpfn_lab import coresets as cs
from tabpfn_lab.experiments import (
    is_transductive,
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


METHODS = ["perest_random", "perest_partition", "builtin_subsample", "perest_stratified", "perest_representative", "perest_kcenter", "local_mix", "cluster_experts", "crumb", "crumb_perest"]


def test_perest_run_flags_timing_analysis_and_determinism(tmp_path):
    csv = run_coreset(_cfg(tmp_path / "a", METHODS))
    assert cli.validate_csv(csv) == []
    df = read_results(csv)
    have = set(df["method"])
    assert {"local_mix_r0.5", "local_mix_r1", "cluster_experts+gate", "crumb", "crumb_perest", "full"} <= have
    loc = df["method"].map(is_transductive)
    assert (df.loc[loc, "flags"] == "transductive").all() and (df.loc[~loc, "flags"] == "").all()
    t = read_timing(timing_path(csv))
    assert set(t["cell"]) == set(df["cell"])
    for m in ("local_mix_r0.5", "crumb", "crumb_perest"):
        assert (t.loc[t["method"] == m, "n_fits"] == 3).all()  # one fit per test group
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


def _mmd2(A_, B_, gamma):
    k = lambda P, Q: np.exp(-gamma * ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).mean()  # noqa: E731
    return k(A_, A_) + k(B_, B_) - 2 * k(A_, B_)


def test_kernel_herding_matches_target_better_than_random():
    rng = np.random.default_rng(0)
    F = np.vstack([rng.normal(0, 1, (900, 2)), rng.normal(4, 0.5, (100, 2))])
    target = rng.normal(4, 0.5, (200, 2))  # the small blob only
    gamma = 0.5
    idx = cs.kernel_herding(F, cs.kernel_mean(F, target, gamma), 40, gamma)
    assert len(idx) == 40 and len(np.unique(idx)) == 40
    np.testing.assert_array_equal(idx, cs.kernel_herding(F, cs.kernel_mean(F, target, gamma), 40, gamma))
    rand = rng.choice(len(F), 40, replace=False)
    assert _mmd2(F[idx], target, gamma) < 0.2 * _mmd2(F[rand], target, gamma)
    assert (idx >= 900).mean() > 0.9


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_crumb_routing(task):
    X, y = make_synthetic(900, 5, n_classes=3, task=task, seed=3)
    Xtr, ytr, Xte = X[:600], y[:600], X[600:]
    faithful = cs.ROUTED_METHODS["crumb"](Xtr, ytr, Xte, 50, 4, 1, task, n_groups=4)
    perest = cs.ROUTED_METHODS["crumb_perest"](Xtr, ytr, Xte, 50, 4, 1, task, n_groups=4)
    local = cs.local_mix_routing(Xtr, ytr, Xte, 50, 4, 1, task, 0.9, n_groups=4)
    assert [r.tolist() for r, _ in faithful] == [r.tolist() for r, _ in perest] == [r.tolist() for r, _ in local]
    assert sorted(np.concatenate([r for r, _ in faithful]).tolist()) == list(range(len(Xte)))
    for (_, a), (_, b) in zip(faithful, perest):
        assert all(np.array_equal(a[0], s) for s in a)  # one shared herded context
        assert len({tuple(sorted(s.tolist())) for s in b}) == 4  # a different one per estimator
        for s in a + b:
            assert len(s) == 50 and len(np.unique(s)) == 50 and s.max() < len(Xtr)
            if task == "classification":
                assert set(np.unique(ytr[s])) == {0, 1, 2}


def test_crumb_never_sees_test_labels(tmp_path, monkeypatch):
    seen = []
    real = experiments.cs.ROUTED_METHODS["crumb"]

    def spy(X_train, y_train, X_test, *a, **kw):
        seen.append((len(X_train), len(y_train), len(X_test)))
        return real(X_train, y_train, X_test, *a, **kw)

    monkeypatch.setitem(experiments.cs.ROUTED_METHODS, "crumb", spy)
    cfg = _cfg(tmp_path, ["perest_random", "crumb"], datasets={"dev": [_cfg(tmp_path, [])["datasets"]["dev"][0]], "confirm": []})
    run_coreset(cfg)
    split = experiments.Context(experiments.normalize_config(cfg, "coreset"), None).split("clf", 0)
    assert seen and all(s == (split.n_train, split.n_train, split.n_test) for s in seen)


# --------------------------------------------------------------------------- gate fitted on held-out training rows


def _gate_toy(rng, n=600, E=4, good=2.0):
    """Row i belongs to expert home[i]; `d` is small to the home centre; `good` is the home expert's log-lik edge."""
    home = rng.integers(0, E, n)
    d = rng.uniform(4, 9, (E, n))
    d[home, np.arange(n)] = rng.uniform(0, 1, n)
    L = rng.normal(-2.0, 0.3, (E, n))
    L[home, np.arange(n)] += good
    return d, L, home


def test_fit_gate_follows_expert_competence():
    rng = np.random.default_rng(0)
    d, L, _ = _gate_toy(rng)
    sharp = cs.fit_gate(d, L, tau0=5.0)
    assert sharp["tau"] < 5.0 and sharp["loss"] < sharp["loss_heuristic"] and sharp["nearest_is_best"] > 0.95
    # the nearest expert is no better than the others: the fitted gate flattens towards the plain mean
    flat = cs.fit_gate(d, rng.normal(-2.0, 0.3, d.shape), tau0=0.5)
    assert flat["tau"] > 0.5 and flat["loss"] <= flat["loss_heuristic"]
    assert sharp["bias"] is None and sharp["loss"] == pytest.approx(cs.gate_loss(d, L, sharp["tau"]))


def test_fit_gate_bias_downweights_a_bad_expert():
    rng = np.random.default_rng(1)
    d, L, home = _gate_toy(rng, good=0.5)
    L[0] -= 3.0  # expert 0 is poor everywhere, on its own cluster too
    plain = cs.fit_gate(d, L, tau0=1.0)
    fit = cs.fit_gate(d, L, tau0=1.0, bias=True)
    assert fit["loss"] < plain["loss"]
    assert fit["bias"][0] == fit["bias"].min() and fit["bias"][0] < fit["bias"][1:].mean() - 1.0
    w = cs.gate_weights(d, fit["tau"], fit["bias"])
    np.testing.assert_allclose(w.sum(axis=0), 1.0)
    assert w[0, home == 0].mean() < cs.gate_weights(d, plain["tau"])[0, home == 0].mean()
    again = cs.fit_gate(d, L, tau0=1.0, bias=True)
    assert again["tau"] == fit["tau"] and np.array_equal(again["bias"], fit["bias"])


def test_gate_weights_without_bias_is_the_heuristic_gate(data):
    X, _ = data
    (d,), tau = cs.cluster_gate_distances(X[:500], [X[500:]], 4, 0)
    assert np.array_equal(cs.gate_weights(d, tau), cs.cluster_gate(X[:500], X[500:], 4, 0))


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_gate_holdout(task):
    y = np.r_[np.zeros(197, int), np.ones(3, int)] if task == "classification" else np.arange(200.0)
    for seed in range(20):
        pool, val = cs.gate_holdout(y, seed, task, val_fraction=0.5)
        assert sorted(np.r_[pool, val].tolist()) == list(range(200)) and 97 <= len(val) <= 100
        if task == "classification":
            assert set(y[pool]) == {0, 1}
    assert np.array_equal(cs.gate_holdout(y, 3, task)[1], cs.gate_holdout(y, 3, task)[1])
    assert len(cs.gate_holdout(y, 3, task, val_max=10)[1]) == 10


def test_fit_gate_regression_minimises_error_of_the_mixture_mean():
    rng = np.random.default_rng(2)
    d, _, home = _gate_toy(rng)
    y = rng.normal(0, 1, d.shape[1])
    mu = y + rng.normal(0, 1.0, d.shape)  # every expert is noisy ...
    mu[home, np.arange(d.shape[1])] = y + rng.normal(0, 0.1, d.shape[1])  # ... except on its own rows
    plain = cs.fit_gate(d, mu, tau0=5.0, y=y)
    assert plain["tau"] < 5.0 and plain["loss"] < plain["loss_heuristic"] and plain["nearest_is_best"] > 0.8
    m = (cs.gate_weights(d, plain["tau"]) * mu).sum(axis=0)
    assert plain["loss"] == pytest.approx(np.mean((m - y) ** 2) / np.var(y))
    mu[0] += 2.0  # expert 0 is biased everywhere, on its own rows too
    fit = cs.fit_gate(d, mu, tau0=5.0, y=y, bias=True)
    assert fit["loss"] < cs.fit_gate(d, mu, tau0=5.0, y=y)["loss"] and fit["bias"][0] == fit["bias"].min()


@pytest.mark.parametrize("regression", [False, True])
def test_fit_gate_gradient(regression, monkeypatch):
    from scipy.optimize import check_grad, minimize

    rng = np.random.default_rng(3)
    d, score = rng.uniform(0, 9, (5, 200)), rng.normal(-2, 1, (5, 200))
    y = rng.normal(0, 1, 200) if regression else None
    errs = []

    def checking(fun, x0, jac=None, **kw):
        for x in (x0, x0 + rng.normal(0, 0.3, len(x0))):
            errs.append(check_grad(lambda t: fun(t)[0], lambda t: fun(t)[1], x))
        return minimize(fun, x0, jac=jac, **kw)

    monkeypatch.setattr("scipy.optimize.minimize", checking)
    cs.fit_gate(d, score, tau0=1.0, y=y, bias=True)
    assert len(errs) == 2 and max(errs) < 1e-5


TUNED = ["cluster_experts_tuned", "cluster_experts_tuned+bias", "cluster_experts_tuned+heuristic"]


@pytest.mark.parametrize("mode", ["single_fit", "separate_fits"])
def test_tuned_gate_run(tmp_path, mode):
    methods = ["perest_random", "cluster_experts", "cluster_experts_tuned"]
    csv = run_coreset(_cfg(tmp_path / "a", methods, perest_mode=mode))
    assert cli.validate_csv(csv) == []
    df = read_results(csv)
    assert set(TUNED) <= set(df["method"])
    assert (df.loc[df["method"].isin(TUNED), "flags"] == "").all()  # a row's weights depend on that row only
    assert df.groupby("method")["cell"].nunique()[TUNED].nunique() == 1
    t = read_timing(timing_path(csv))
    fits = 2 if mode == "single_fit" else 8  # the experts are fitted again to predict the held-out training rows
    assert (t.loc[t["method"] == "cluster_experts_tuned", "n_fits"] == fits).all()
    dev = cli.analyze(csv)["dev"]
    assert dev.loc[dev["method"].isin(TUNED), "improvement"].notna().all()
    again = run_coreset(_cfg(tmp_path / "b", methods, perest_mode=mode))
    assert csv.read_bytes() == again.read_bytes()


@pytest.mark.parametrize("ds", [0, 1])
def test_tuned_gate_never_uses_test_labels(tmp_path, monkeypatch, ds):
    """Scrambling the test labels changes neither the held-out slice, the labels the gate is fitted on, nor the fit."""
    import dataclasses

    spec = _cfg(tmp_path, [])["datasets"]["dev"][ds]
    cfg = experiments.normalize_config(_cfg(tmp_path, ["cluster_experts_tuned"], datasets={"dev": [spec], "confirm": []}), "coreset")
    ctx = experiments.Context(cfg, None)
    split = ctx.split(spec["name"], 0)
    calls = []
    real = cs.fit_gate

    def spy(d, score, tau0, y=None, **kw):
        res = real(d, score, tau0, y=y, **kw)
        assert (y is None) == (split.task == "classification")
        assert y is None or np.array_equal(y, split.y_train[cs.gate_holdout(split.y_train, 0, split.task)[1]])
        calls.append((np.array(d), np.array(score), res["tau"], res["bias"]))
        return res

    monkeypatch.setattr(experiments.cs, "fit_gate", spy)
    experiments.evaluate_coreset(cfg, ctx, split, "cluster_experts_tuned", 60, 0)
    y_bad = np.random.default_rng(0).permutation(split.y_test)
    assert not np.array_equal(y_bad, split.y_test)
    experiments.evaluate_coreset(cfg, ctx, dataclasses.replace(split, y_test=y_bad), "cluster_experts_tuned", 60, 0)
    assert len(calls) == 4
    n_val = len(cs.gate_holdout(split.y_train, 0, split.task)[1])
    for a, b in zip(calls[:2], calls[2:]):
        assert a[0].shape == a[1].shape == (4, n_val)  # fitted on the held-out training slice
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
        assert (a[3] is None and b[3] is None) or np.array_equal(a[3], b[3])


def test_tuned_gate_holdout_rows_are_in_no_context(tmp_path, monkeypatch):
    spec = _cfg(tmp_path, [])["datasets"]["dev"][0]
    cfg = experiments.normalize_config(_cfg(tmp_path, ["cluster_experts_tuned"], datasets={"dev": [spec], "confirm": []}), "coreset")
    ctx = experiments.Context(cfg, None)
    split = ctx.split("clf", 0)
    seen = []
    real = ctx.outputs

    def spy(split_, X, y, E, seed, tag, overrides=None, X_eval=None):
        seen.append((X, X_eval, overrides["SUBSAMPLE_SAMPLES"]))
        return real(split_, X, y, E, seed, tag, overrides=overrides, X_eval=X_eval)

    monkeypatch.setattr(ctx, "outputs", spy)
    experiments.evaluate_coreset(cfg, ctx, split, "cluster_experts_tuned", 60, 0)
    pool, val = cs.gate_holdout(split.y_train, 0, split.task)
    (X_fit, ev_test, subs), (X_fit2, ev_val, subs2) = seen
    assert np.array_equal(X_fit, split.X_train[pool]) and np.array_equal(X_fit2, X_fit) and ev_test is None
    assert np.array_equal(ev_val, split.X_train[val])
    assert all(np.array_equal(a, b) and len(a) == 60 and a.max() < len(pool) for a, b in zip(subs, subs2))
