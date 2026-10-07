import numpy as np
import pandas as pd
import pytest

from tabpfn_lab import aggregation as A
from tabpfn_lab import stats
from tabpfn_lab.experiments import CANARY, read_results, run_aggregation, run_coreset

from conftest import SpyBackend

CLF_DS = [
    {"name": f"syn{i}", "source": "synthetic", "task": "classification", "n": 360, "imbalance": [0.75, 0.15, 0.1], "noise": 0.05, "data_seed": 10 + i}
    for i in range(3)
]


def _coreset(smoke, **kw):
    base = dict(datasets={"dev": [], "confirm": CLF_DS}, budget_fractions=[0.5, 0.2], selection_seeds=[0, 1], n_estimators=4, learning_curve_points=6)
    base.update(kw)
    return read_results(run_coreset(smoke(**base), backend=SpyBackend()))


@pytest.mark.slow
def test_null_experiment(smoke):
    # random vs random with different selection seeds: relabel seed s+1 rows as "random_b" paired with seed s
    df = _coreset(smoke, coreset_methods=["random"], selection_seeds=list(range(6)), budget_fractions=[0.2, 0.1])
    df = df[df["budget_key"] != "lc"]
    a = df[df["seed"] % 2 == 0].copy()
    b = df[df["seed"] % 2 == 1].copy()
    b["seed"] -= 1
    b["method"] = "random_b"
    both = pd.concat([a, b])
    for metric in ("log_loss", "accuracy"):
        for key in ("f0.2", "f0.1"):
            r = stats.compare(both[both["budget_key"] == key], "random_b", "random", metric)
            assert not r["significant"], r


def test_canary_bad_baseline(smoke):
    df = _coreset(smoke, coreset_methods=["random", "stratified", "kmeans_stratified", CANARY], selection_seeds=[0])
    ll = df[(df["metric"] == "log_loss") & ~df["budget_key"].isin(["lc", "full"])]
    per = ll.groupby(["dataset", "budget", "method"])["value"].mean().unstack("method")
    for m in ("random", "stratified", "kmeans_stratified"):
        assert (per[m] < per[CANARY]).all(), per


def test_full_budget_equals_full(smoke):
    from tabpfn_lab.experiments import Context, evaluate_coreset, normalize_config

    cfg = normalize_config(smoke(datasets={"dev": [], "confirm": CLF_DS[:1]}, n_estimators=3), "coreset")
    ctx = Context(cfg, SpyBackend())
    split = ctx.split("syn0", 0)
    full = evaluate_coreset(cfg, ctx, split, "full", split.n_train, 0)[0][1]
    for m in ("random", "stratified", "kmeans_stratified", "uncertainty_mix"):
        for b in (split.n_train, split.n_train + 10):
            assert evaluate_coreset(cfg, ctx, split, m, b, 0)[0][1] == full, m


@pytest.mark.slow
def test_more_data_helps_on_average(smoke):
    df = _coreset(smoke, coreset_methods=["random"], selection_seeds=[0, 1, 2])
    ll = df[(df["metric"] == "log_loss") & (df["method"] == "random")]
    per = ll.groupby(["dataset", "budget"])["value"].mean().reset_index()
    big = per.loc[per.groupby("dataset")["budget"].idxmax(), "value"].mean()
    small = per.loc[per.groupby("dataset")["budget"].idxmin(), "value"].mean()
    assert big <= small + 0.02


@pytest.mark.slow
def test_ensembling_helps_or_ties(smoke):
    cfg = smoke(datasets=CLF_DS, seeds=[0, 1, 2], n_estimators=[1, 8], aggregators=["mean"], references=[])
    df = read_results(run_aggregation(cfg, backend=SpyBackend()))
    ll = df[(df["metric"] == "log_loss") & (df["n_estimators"] == 8)]
    mean8 = ll[ll["method"] == "mean"].groupby("dataset")["value"].mean().mean()
    single = ll[ll["method"] == "single"].groupby("dataset")["value"].mean().mean()
    assert mean8 <= single + 0.02


def test_aggregator_equals_mean_when_degenerate(smoke):
    aggs = [a for a in A.CLF_AGGREGATORS if a not in A.CHANGES_SINGLE_ESTIMATOR and a != "weighted"]
    cfg = smoke(datasets=CLF_DS[:2], seeds=[0], n_estimators=[1], aggregators=aggs, references=[])
    df = read_results(run_aggregation(cfg, backend=SpyBackend()))
    single = df[df["method"] == "single"].set_index(["dataset", "metric"])["value"]
    for a in aggs:
        got = df[df["method"] == a].set_index(["dataset", "metric"])["value"]
        pd.testing.assert_series_equal(got, single, check_names=False)


def test_no_nan_or_inf_in_results(smoke):
    agg = read_results(run_aggregation(smoke(), backend=SpyBackend()))
    assert np.isfinite(agg["value"]).all()
    cor = read_results(run_coreset(smoke(out_csv=smoke()["out_csv"].replace(".csv", "_b.csv")), backend=SpyBackend()))
    assert np.isfinite(cor["value"]).all()
    from tabpfn_lab.cli import validate_csv
    from pathlib import Path

    assert validate_csv(Path(smoke()["out_csv"])) == []
    assert validate_csv(Path(smoke()["out_csv"].replace(".csv", "_b.csv"))) == []


def test_regression_path_runs(smoke):
    ds = [{"name": "syn_reg", "source": "synthetic", "task": "regression", "n": 300, "data_seed": 4}]
    df = read_results(run_aggregation(smoke(datasets=ds, seeds=[0]), backend=SpyBackend()))
    assert set(df["metric"]) == {"rmse", "crps"}
    assert set(df["method"]) >= {"single", "mixture", "log_pool", "quantile_mean", "quantile_median", "native"}
    assert np.isfinite(df["value"]).all()
