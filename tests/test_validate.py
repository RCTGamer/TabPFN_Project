import json

import numpy as np
import pytest

from tabpfn_lab.cli import validate_csv
from tabpfn_lab.experiments import read_results, run_aggregation, run_coreset
from tabpfn_lab.validate import validate_results

from conftest import SpyBackend

DS = [
    {"name": "syn_a", "source": "synthetic", "task": "classification", "n": 300, "imbalance": [0.8, 0.12, 0.08], "noise": 0.02, "data_seed": 1},
    {"name": "syn_r", "source": "synthetic", "task": "regression", "n": 300, "data_seed": 2},
]


@pytest.fixture(scope="module")
def coreset_run(tmp_path_factory):
    from conftest import load_smoke

    tmp = tmp_path_factory.mktemp("validate")
    cfg = load_smoke(
        tmp,
        datasets={"dev": [], "confirm": DS},
        coreset_methods=["random", "stratified", "builtin_subsample", "builtin_majority_downsample"],
        budget_fractions=[0.5, 0.3],
        selection_seeds=[0, 1],
        n_estimators=2,
    )
    csv = run_coreset(cfg, backend=SpyBackend())
    meta = json.loads(csv.with_name(csv.stem + ".meta.json").read_text())
    return read_results(csv), meta, csv


def _check(df, meta):
    e = meta["expected"]
    return validate_results(df, e["methods"], e["datasets"], e["seeds"], e["budgets"], required_baselines=e["required_baselines"], min_random_budgets=6)


def test_clean_dataframe_is_clean(coreset_run):
    df, meta, csv = coreset_run
    assert _check(df, meta) == []
    assert validate_csv(csv) == []


def test_clean_aggregation_run(smoke):
    cfg = smoke(datasets=DS, seeds=[0, 1], n_estimators=[1, 2])
    csv = run_aggregation(cfg, backend=SpyBackend())
    assert validate_csv(csv) == []


def _has(problems, text):
    return any(text in p for p in problems), problems


def test_missing_column(coreset_run):
    df, meta, _ = coreset_run
    assert _has(_check(df.drop(columns=["tier"]), meta), "missing columns")[0]


def test_wrong_dtype(coreset_run):
    df, meta, _ = coreset_run
    bad = df.copy()
    bad["seed"] = bad["seed"].astype(str)
    assert _has(_check(bad, meta), "dtype")[0]


def test_unknown_metric(coreset_run):
    df, meta, _ = coreset_run
    bad = df.copy()
    bad.loc[bad.index[0], "metric"] = "f1_magic"
    assert _has(_check(bad, meta), "unknown metric")[0]


@pytest.mark.parametrize("v", [np.nan, np.inf])
def test_nan_or_inf_value(coreset_run, v):
    df, meta, _ = coreset_run
    bad = df.copy()
    bad.loc[bad.index[3], "value"] = v
    assert _has(_check(bad, meta), "NaN/inf")[0]


def test_duplicate_rows(coreset_run):
    df, meta, _ = coreset_run
    import pandas as pd

    bad = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    assert _has(_check(bad, meta), "duplicate")[0]


@pytest.mark.parametrize("metric,value", [("accuracy", 1.2), ("roc_auc", -0.1), ("ece", 1.5), ("log_loss", -0.2), ("rmse", -1.0), ("crps", -0.5)])
def test_out_of_range(coreset_run, metric, value):
    df, meta, _ = coreset_run
    bad = df.copy()
    i = bad.index[bad["metric"] == metric][0]
    bad.loc[i, "value"] = value
    assert _has(_check(bad, meta), f"out-of-range values for {metric}")[0]


@pytest.mark.parametrize(
    "drop,text",
    [
        (lambda d: d["method"] == "stratified", "method 'stratified' missing"),
        (lambda d: d["seed"] == 1, "seed 1 missing"),
        (lambda d: d["dataset"] == "syn_r", "dataset 'syn_r' missing"),
        (lambda d: (d["method"] == "stratified") & (d["budget_key"] == "f0.3"), "missing budgets"),
    ],
)
def test_incomplete_grid(coreset_run, drop, text):
    df, meta, _ = coreset_run
    assert _has(_check(df[~drop(df)], meta), text)[0]


def test_budget_larger_than_n_train(coreset_run):
    df, meta, _ = coreset_run
    bad = df.copy()
    i = bad.index[bad["method"] == "stratified"][0]
    bad.loc[i, "budget"] = bad.loc[i, "n_train"] + 5
    assert _has(_check(bad, meta), "budget > n_train")[0]
