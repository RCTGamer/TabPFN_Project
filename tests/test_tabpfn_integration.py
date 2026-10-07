"""Real-model tests (@pytest.mark.tabpfn): run on the GPU server, skipped without tabpfn + CUDA.

The last three tests (sharding, resume, failure recording) use the sklearn backend and always run.
"""

import numpy as np
import pandas as pd
import pytest

from tabpfn_lab import aggregation as A
from tabpfn_lab.backends import BarDistribution, softmax
from tabpfn_lab.cli import merge, validate_csv
from tabpfn_lab.experiments import read_results, run_coreset
from tabpfn_lab.sizes import make_synthetic

from conftest import SpyBackend

ATOL = 1e-4  # GPU nondeterminism tolerance


@pytest.fixture(scope="module")
def tb():
    from tabpfn_lab.backends import TabPFNBackend

    return TabPFNBackend(version="3.5", device="cuda", inference_precision="float32")


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_synthetic(500, 6, n_classes=3, seed=0, noise=0.05)
    return X[:350], y[:350], X[350:], y[350:]


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_synthetic(500, 5, task="regression", seed=1, noise=0.1)
    return X[:350], y[:350], X[350:], y[350:]


def _stock(task, E, seed, **kw):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
    return cls.create_default_for_version(ModelVersion.V3_5, n_estimators=E, random_state=seed, device="cuda", inference_precision=__import__("torch").float32, **kw)


@pytest.mark.tabpfn
def test_shapes_and_normalisation(tb, clf_data):
    Xtr, ytr, Xte, _ = clf_data
    out = tb.clf_outputs(Xtr, ytr, Xte, np.arange(3), 4, 0)
    assert out.logits.shape == (4, len(Xte), 3) and np.isfinite(out.logits).all()
    np.testing.assert_allclose(A.agg_mean(out.logits, out.temperature).sum(1), 1, atol=1e-6)


@pytest.mark.tabpfn
def test_estimators_actually_differ_and_random_state_controls(tb, clf_data):
    Xtr, ytr, Xte, _ = clf_data
    a = tb.clf_outputs(Xtr, ytr, Xte, np.arange(3), 4, 0)
    b = tb.clf_outputs(Xtr, ytr, Xte, np.arange(3), 4, 1)
    a2 = tb.clf_outputs(Xtr, ytr, Xte, np.arange(3), 4, 0)
    assert not np.allclose(a.logits[0], a.logits[1])  # members differ within one fit
    assert not np.allclose(a.logits, b.logits)  # random_state changes the members
    np.testing.assert_allclose(a.logits, a2.logits, atol=ATOL)  # and repeats


@pytest.mark.tabpfn
def test_accuracy_above_trivial(tb, clf_data):
    Xtr, ytr, Xte, yte = clf_data
    out = tb.clf_outputs(Xtr, ytr, Xte, np.arange(3), 4, 0)
    acc = (A.agg_mean(out.logits, out.temperature).argmax(1) == yte).mean()
    assert acc > np.bincount(yte).max() / len(yte)


@pytest.mark.tabpfn
def test_regression_outputs_valid(tb, reg_data):
    Xtr, ytr, Xte, yte = reg_data
    out = tb.reg_outputs(Xtr, ytr, Xte, 4, 0)
    E, n, B = out.probs.shape
    assert (E, n) == (4, len(Xte)) and np.isfinite(out.probs).all()
    np.testing.assert_allclose(out.probs.sum(-1), 1, atol=1e-4)
    pred = A.agg_mixture(out.probs, out.dist)
    assert (np.diff(pred.quantiles, axis=1) >= -1e-6).all()
    med = out.dist.icdf(pred.logp, 0.5)
    assert np.corrcoef(med, yte)[0, 1] > 0.8


@pytest.mark.tabpfn
def test_coreset_missing_class_ok(tb, clf_data):
    Xtr, ytr, Xte, _ = clf_data
    keep = ytr != 1
    out = tb.clf_outputs(Xtr[keep], ytr[keep], Xte, np.arange(3), 2, 0)
    P = A.agg_mean(out.logits, out.temperature)
    assert P.shape == (len(Xte), 3) and P[:, 1].max() < 1e-6


@pytest.mark.tabpfn
def test_seed_reproducible(tb, reg_data):
    Xtr, ytr, Xte, _ = reg_data
    np.testing.assert_allclose(tb.reg_outputs(Xtr, ytr, Xte, 2, 3).probs, tb.reg_outputs(Xtr, ytr, Xte, 2, 3).probs, atol=ATOL)


@pytest.mark.tabpfn
def test_baseline_reproduces_native_clf(clf_data):
    Xtr, ytr, Xte, _ = clf_data
    clf = _stock("classification", 4, 0).fit(Xtr, ytr)
    raw = clf.predict_raw_logits(Xte)
    np.testing.assert_allclose(A.agg_mean(raw, clf.softmax_temperature_), clf.predict_proba(Xte), atol=ATOL)
    geo = _stock("classification", 4, 0, average_before_softmax=True).fit(Xtr, ytr)
    np.testing.assert_allclose(A.agg_logit_mean(raw, clf.softmax_temperature_), geo.predict_proba(Xte), atol=ATOL)


@pytest.mark.tabpfn
def test_baseline_reproduces_native_reg(tb, reg_data):
    Xtr, ytr, Xte, _ = reg_data
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    out = tb.reg_outputs(Xtr, ytr, Xte, 4, 0)
    nat = tb.native_reg(Xtr, ytr, Xte, 4, 0, levels)
    pred = A.agg_mixture(out.probs, out.dist, levels)
    scale = np.std(ytr)
    np.testing.assert_allclose(pred.mean, nat["mean"], atol=1e-3 * scale)
    np.testing.assert_allclose(out.dist.icdf(pred.logp, 0.5), nat["median"], atol=1e-3 * scale)
    np.testing.assert_allclose(pred.quantiles, nat["quantiles"], atol=1e-3 * scale)
    geo = _stock("regression", 4, 0, average_before_softmax=True).fit(Xtr, ytr)
    np.testing.assert_allclose(A.agg_log_pool(out.probs, out.dist, levels).mean, geo.predict(Xte), atol=1e-3 * scale)


@pytest.mark.tabpfn
def test_class_order_aligned(clf_data):
    Xtr, ytr, Xte, _ = clf_data
    for labels in (np.array(["b", "c", "a"])[ytr], np.array([5, 11, 2])[ytr]):
        clf = _stock("classification", 2, 0).fit(Xtr, labels)
        raw = clf.predict_raw_logits(Xte)
        assert raw.shape[-1] == len(clf.classes_)
        P = A.agg_mean(raw, clf.softmax_temperature_)
        np.testing.assert_array_equal(clf.classes_[P.argmax(1)], clf.predict(Xte))


@pytest.mark.tabpfn
def test_n_estimators_recorded_and_config_saved(tb, clf_data):
    import tabpfn

    Xtr, ytr, Xte, _ = clf_data
    out = tb.clf_outputs(Xtr, ytr, Xte, np.arange(3), 4, 0)
    assert out.meta["n_estimators_"] == 4
    assert out.meta["tabpfn_version"] == tabpfn.__version__
    assert out.meta["model_version"] == "3.5"
    assert isinstance(out.meta["inference_config"], dict) and "SUBSAMPLE_SAMPLES" in out.meta["inference_config"]
    rng = np.random.default_rng(0)
    wide = tb.clf_outputs(rng.normal(size=(200, 600)), ytr[:200], rng.normal(size=(20, 600)), np.arange(3), 4, 0)
    assert wide.meta["n_estimators_"] >= 1 and wide.logits.shape[0] == wide.meta["n_estimators_"]


@pytest.mark.tabpfn
def test_builtin_subsample_baseline_runs(tb, clf_data):
    Xtr, ytr, Xte, _ = clf_data
    methods = tb.subsample_methods()
    from tabpfn.preprocessing.configs import SampleSubsamplingMethod

    assert methods == [m.value for m in SampleSubsamplingMethod]
    full = tb.clf_outputs(Xtr, ytr, Xte, np.arange(3), 4, 0)
    sub = tb.clf_outputs(Xtr, ytr, Xte, np.arange(3), 4, 0, overrides={"SUBSAMPLE_SAMPLES": 50, "SAMPLE_SUBSAMPLING_METHOD": methods[0]})
    assert not np.allclose(full.logits, sub.logits)


@pytest.mark.tabpfn
def test_embedding_shapes(clf_data):
    Xtr, ytr, Xte, _ = clf_data
    clf = _stock("classification", 2, 0).fit(Xtr, ytr)
    emb = clf.get_embeddings(Xte, data_source="test")
    assert emb.ndim == 3 and emb.shape[:2] == (2, len(Xte))
    cached = _stock("classification", 2, 0, fit_mode="fit_with_cache").fit(Xtr, ytr)
    with pytest.raises(Exception, match="fit_with_cache"):
        cached.get_embeddings(Xtr, data_source="train")


@pytest.mark.tabpfn
def test_batched_matches_unbatched(clf_data):
    Xtr, ytr, Xte, _ = clf_data
    clf = _stock("classification", 2, 0)
    halves = [(Xtr[:170].astype(np.float32), ytr[:170]), (Xtr[170:340].astype(np.float32), ytr[170:340])]
    if not all(set(h[1]) == {0, 1, 2} for h in halves):
        pytest.skip("halves must share the class set")
    batched = clf.predict_proba_batched([h[0] for h in halves], [h[1] for h in halves], [Xte.astype(np.float32)] * 2)
    for h, pb in zip(halves, batched):
        single = _stock("classification", 2, 0).fit(*h).predict_proba(Xte.astype(np.float32))
        np.testing.assert_allclose(pb, single, atol=1e-3)


# ---------------------------------------------------------------- runner robustness (sklearn backend, always run)

DS = [{"name": "syn", "source": "synthetic", "task": "classification", "n": 300, "imbalance": [0.8, 0.12, 0.08], "noise": 0.02, "data_seed": 3}]


def _cfg(smoke, **kw):
    base = dict(
        datasets={"dev": [], "confirm": DS},
        coreset_methods=["random", "stratified", "builtin_subsample", "builtin_majority_downsample"],
        budget_fractions=[0.5, 0.3],
        selection_seeds=[0, 1],
        n_estimators=2,
    )
    base.update(kw)
    return smoke(**base)


class OOMBackend(SpyBackend):
    """Raises an OOM-shaped error for one coreset method."""

    def clf_outputs(self, X_train, y_train, X_test, classes, n_estimators, seed, overrides=None):
        if len(X_train) == 63 and seed == 1:  # stratified/random at f0.3, selection seed 1
            raise type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory. Tried to allocate 2.00 GiB")
        return super().clf_outputs(X_train, y_train, X_test, classes, n_estimators, seed, overrides)


def test_oom_is_recorded_not_fatal(smoke):
    csv = run_coreset(_cfg(smoke, cache_dir=None), backend=OOMBackend())
    fails = pd.read_csv(csv.with_name(csv.stem + ".failures.csv"))
    assert len(fails) >= 1 and fails["oom"].all()
    df = read_results(csv)
    assert len(df) and set(df["cell"]).isdisjoint(set(fails["cell"]))
    problems = validate_csv(csv)
    assert any("failed cell" in p for p in problems)
    assert any("missing" in p for p in problems)


def test_shard_union_equals_full_grid(smoke, tmp_path):
    full = read_results(run_coreset(_cfg(smoke, out_csv=str(tmp_path / "full" / "x.csv")), backend=SpyBackend()))
    out = tmp_path / "sh" / "x.csv"
    for i in range(3):
        run_coreset(_cfg(smoke, out_csv=str(out)), backend=SpyBackend(), shard=(i, 3))
    merged_csv, problems = merge(out)
    assert problems == []
    merged = read_results(merged_csv)
    key = ["cell", "method", "metric"]
    a = full.sort_values(key).reset_index(drop=True)
    b = merged.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


class Killed(BaseException):
    pass


class DyingBackend(SpyBackend):
    def __init__(self, die_after, **kw):
        super().__init__(**kw)
        self.die_after = die_after

    def clf_outputs(self, *a, **kw):
        if len(self.calls) >= self.die_after:
            raise Killed()
        return super().clf_outputs(*a, **kw)


def test_resume_skips_finished_cells(smoke):
    cfg = _cfg(smoke, cache_dir=None)
    with pytest.raises(Killed):
        run_coreset(cfg, backend=DyingBackend(die_after=5))
    from pathlib import Path

    partial = read_results(Path(cfg["out_csv"]))
    done = set(partial["cell"])
    assert 0 < len(done) < 20
    be = SpyBackend()
    csv = run_coreset(cfg, backend=be)
    df = read_results(csv)
    n_cells = df["cell"].nunique()
    assert len(be.calls) == n_cells - len(done)  # each remaining cell is one backend fit; finished ones are skipped
    assert not df.duplicated(["cell", "method", "metric"]).any()
    assert validate_csv(csv) == []
