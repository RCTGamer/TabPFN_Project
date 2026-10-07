import numpy as np
import pandas as pd
import pytest

from tabpfn_lab import coresets as cs
from tabpfn_lab.datasets import make_split
from tabpfn_lab.experiments import read_results, run_aggregation, run_coreset
from tabpfn_lab.sizes import make_synthetic

from conftest import SpyBackend

SMALL = [{"name": "syn", "source": "synthetic", "task": "classification", "n": 300, "imbalance": [0.8, 0.12, 0.08], "noise": 0.02, "data_seed": 5}]


def _coreset_cfg(smoke, **kw):
    base = dict(
        datasets={"dev": [], "confirm": SMALL},
        coreset_methods=["random", "stratified", "builtin_subsample", "builtin_majority_downsample"],
        budget_fractions=[0.5],
        selection_seeds=[0, 1],
        n_estimators=2,
    )
    base.update(kw)
    return smoke(**base)


def test_split_disjoint_and_complete():
    X, y = make_synthetic(500, 4, n_classes=3, seed=0)
    s = make_split(X, y, "classification", 0.3, seed=1)
    assert not set(s.train_idx) & set(s.test_idx)
    assert set(s.train_idx) | set(s.test_idx) == set(range(500))
    full = np.bincount(y) / len(y)
    np.testing.assert_allclose(np.bincount(s.y_test) / s.n_test, full, atol=0.02)


def test_same_split_across_methods(smoke):
    cfg = _coreset_cfg(smoke)
    be = SpyBackend()
    csv = run_coreset(cfg, backend=be)
    # every fit evaluates on the same number of test rows, and the split hash is recorded once per (dataset, split_seed)
    assert len({n_test for _, _, n_test in be.calls}) == 1
    import json

    meta = json.loads(csv.with_name(csv.stem + ".meta.json").read_text())
    assert list(meta["split_hashes"]) == ["syn|0"]


def test_coreset_selected_from_train_only(smoke, monkeypatch):
    cfg = _coreset_cfg(smoke, coreset_methods=["random", "stratified", "kmeans_stratified"])
    seen = []
    real = cs.select_coreset

    def spy(method, X, y, budget, seed, task="classification", **kw):
        idx = real(method, X, y, budget, seed, task, **kw)
        seen.append((len(X), idx.max()))
        return idx

    monkeypatch.setattr(cs, "select_coreset", spy)
    run_coreset(cfg, backend=SpyBackend())
    n_train = int(round(300 * 0.7))
    assert seen and all(n == n_train and m < n_train for n, m in seen)


def test_budget_not_above_n_train(smoke):
    cfg = _coreset_cfg(smoke, budget_fractions=[0.5, 1.5], budget_absolute=[10_000])
    df = read_results(run_coreset(cfg, backend=SpyBackend()))
    over = df[df["method"] != "full"]
    assert (over["budget"] < over["n_train"]).all()
    full = df[df["method"] == "full"]
    assert (full["budget"] == full["n_train"]).all() and len(full)


def test_seed_reproducibility(smoke, tmp_path):
    cfg = _coreset_cfg(smoke, cache_dir=None)
    a = pd.read_csv(run_coreset(dict(cfg, out_csv=str(tmp_path / "a.csv")), backend=SpyBackend()))
    b = pd.read_csv(run_coreset(dict(cfg, out_csv=str(tmp_path / "b.csv")), backend=SpyBackend()))
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    X, y = make_synthetic(300, 4, seed=0)
    s0, s1 = make_split(X, y, "classification", 0.3, seed=0), make_split(X, y, "classification", 0.3, seed=1)
    assert s0.test_hash != s1.test_hash


def test_different_seeds_give_different_metrics(smoke):
    cfg = smoke(datasets=SMALL, seeds=[0, 1], aggregators=["mean"], references=[], n_estimators=2)
    df = read_results(run_aggregation(cfg, backend=SpyBackend()))
    v = df[(df["method"] == "mean") & (df["metric"] == "log_loss")].set_index("seed")["value"]
    assert v.loc[0] != v.loc[1]


def test_backend_called_with_correct_shapes_and_no_test_labels(smoke):
    cfg = smoke(datasets=SMALL, seeds=[0], aggregators=["mean", "val_weighted"], references=["native"], n_estimators=2)
    from tabpfn_lab.datasets import load_split

    s = load_split(SMALL[0], 0, 0.3)
    be = SpyBackend(forbidden_y=[s.y_test])
    run_aggregation(cfg, backend=be)  # SpyBackend asserts shapes, class order, and y_test absence on every call
    assert {c[0] for c in be.calls} >= {"clf_outputs", "native_clf"}

    cfg_b = _coreset_cfg(smoke, out_csv=cfg["out_csv"].replace(".csv", "_b.csv"))
    be_b = SpyBackend(forbidden_y=[s.y_test])
    run_coreset(cfg_b, backend=be_b)
    assert be_b.calls
