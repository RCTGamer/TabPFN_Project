"""tabpfn's per-estimator row subsampling (SUBSAMPLE_SAMPLES / SAMPLE_SUBSAMPLING_METHOD), as emulated by
SklearnBackend, and the experiment-B plumbing around it (builtin_* rows, single-fit per-estimator subsets)."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from tabpfn_lab import coresets as cs
from tabpfn_lab.backends import SklearnBackend, emulate_subsample
from tabpfn_lab.experiments import Context, coreset_family, coreset_methods_by_task, normalize_config


@pytest.fixture(scope="module")
def y3():
    return np.repeat([0, 1, 2], [700, 200, 100])


def test_balanced_round_robin_covers_every_row(y3):
    subs = emulate_subsample(y3, 250, "balanced", 8, np.random.default_rng(0), "classification")
    assert len(subs) == 8 and all(len(s) == 250 and len(np.unique(s)) == 250 for s in subs)
    counts = np.bincount(np.concatenate(subs), minlength=len(y3))
    assert counts.min() >= 1 and counts.max() - counts.min() <= 1  # 2000 slots over 1000 rows: each row twice


def test_stratified_keeps_prior_with_one_per_class(y3):
    subs = emulate_subsample(y3, 50, "auto", 4, np.random.default_rng(0), "classification")
    for s in subs:
        np.testing.assert_array_equal(np.bincount(y3[s], minlength=3), [34, 10, 6])  # 1 + floor(47 * share), largest remainder
    tiny = np.repeat([0, 1, 2], [990, 9, 1])
    for s in emulate_subsample(tiny, 10, "stratified", 3, np.random.default_rng(1), "classification"):
        assert set(tiny[s]) == {0, 1, 2}


def test_majority_downsample_keeps_minorities(y3):
    subs = emulate_subsample(y3, 400, "majority_downsample", 4, np.random.default_rng(0), "classification")
    for s in subs:
        assert (y3[s] != 0).sum() == 300 and len(s) == 400
    with pytest.raises(ValueError):
        emulate_subsample(y3, 300, "majority_downsample", 2, np.random.default_rng(0), "classification")


def test_stratified_rejected_for_regression():
    with pytest.raises(ValueError):
        emulate_subsample(np.linspace(0, 1, 100), 10, "stratified", 2, np.random.default_rng(0), "regression")
    assert len(emulate_subsample(np.linspace(0, 1, 100), 10, "auto", 2, np.random.default_rng(0), "regression")) == 2


def test_no_subsampling_when_budget_covers_data(y3):
    assert emulate_subsample(y3, len(y3), "auto", 4, np.random.default_rng(0), "classification") is None


def test_explicit_index_lists_fit_each_estimator_on_its_rows():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 4)), rng.integers(0, 2, 200)
    subsets = cs.per_estimator_subsets("perest_random", X, y, 60, 3, 5)
    be = SklearnBackend(n_trees=5)
    out = be.clf_outputs(X, y, X[:30], np.arange(2), 3, 4, overrides={"SUBSAMPLE_SAMPLES": subsets})
    for e, s in enumerate(subsets):
        rf = RandomForestClassifier(n_estimators=5, random_state=4 * 1000 + e, n_jobs=1).fit(X[s], y[s])
        p = rf.predict_proba(X[:30]) * (1 - 2 * be.eps) + be.eps
        np.testing.assert_allclose(out.logits[e], np.log(p))


def _cfg(smoke, **kw):
    return normalize_config(smoke(budget_fractions=[0.3], selection_seeds=[0], **kw), "coreset")


def test_builtin_balanced_is_classification_only(smoke):
    cfg = _cfg(smoke, coreset_methods=["random", "builtin_subsample", "builtin_balanced", "builtin_majority_downsample"])
    by_task = coreset_methods_by_task(cfg)
    assert "builtin_balanced" in by_task["classification"] and "builtin_balanced" not in by_task["regression"]


def test_family_against_builtin_baseline(smoke):
    methods = ["random", "representative", "perest_random", "builtin_subsample", "builtin_majority_downsample"]
    cfg = _cfg(smoke, coreset_methods=methods, baseline="builtin_subsample")
    fam = {m for m, _, _ in coreset_family(cfg, Context(cfg, SklearnBackend()))}
    assert fam == {"random", "representative", "perest_random", "builtin_majority_downsample"}
    default = _cfg(smoke, coreset_methods=methods)
    assert {m for m, _, _ in coreset_family(default, Context(default, SklearnBackend()))} == {"representative", "perest_random"}
    bad = _cfg(smoke, coreset_methods=["random"], baseline="builtin_subsample")
    with pytest.raises(ValueError):
        coreset_family(bad, Context(bad, SklearnBackend()))
