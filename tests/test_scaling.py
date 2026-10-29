import numpy as np
import pandas as pd
import pytest

from tabpfn_lab import cli
from tabpfn_lab.experiments import nested_random_subset, read_results, read_timing, run_scaling, scaling_budgets, timing_path


def _cfg(tmp_path, **kw):
    cfg = {
        "name": "scaling_t",
        "backend": {"name": "sklearn", "n_trees": 5},
        "datasets": {
            "dev": [
                {"name": "clf", "source": "synthetic", "task": "classification", "n": 500, "n_features": 5, "n_classes": 3, "imbalance": [0.6, 0.3, 0.1], "noise": 0.02, "data_seed": 1},
                {"name": "reg", "source": "synthetic", "task": "regression", "n": 300, "n_features": 4, "data_seed": 2},
            ],
            "confirm": [],
        },
        "seeds": [0, 1],
        "budget_start": 64,
        "n_estimators": [1, 2, 4],
        "cache_dir": None,
        "log_dir": str(tmp_path / "logs"),
        "out_csv": str(tmp_path / "tables" / "scaling_t.csv"),
    }
    cfg.update(kw)
    return cfg


def test_budgets_double_then_full():
    assert scaling_budgets(5000) == [512, 1024, 2048, 4096, 5000]
    assert scaling_budgets(1024) == [512, 1024]
    assert scaling_budgets(300) == [300]
    assert scaling_budgets(5000, max_budget=2048) == [512, 1024, 2048]
    assert scaling_budgets(100, start=10, factor=3) == [10, 30, 90, 100]
    with pytest.raises(ValueError):
        scaling_budgets(100, factor=1)


def test_nested_subsets_within_a_seed():
    n = 1000
    prev = set()
    for b in scaling_budgets(n, start=64):
        idx = nested_random_subset(n, b, seed=3)
        assert len(idx) == len(np.unique(idx)) == b and idx.min() >= 0 and idx.max() < n
        assert np.all(np.diff(idx) > 0)
        assert prev <= set(idx)
        prev = set(idx)
    assert not np.array_equal(nested_random_subset(n, 64, 0), nested_random_subset(n, 64, 1))


def test_scaling_grid_timing_and_resume(tmp_path):
    cfg = _cfg(tmp_path)
    csv = run_scaling(cfg)
    assert cli.validate_csv(csv) == []
    df = read_results(csv)
    timing = read_timing(timing_path(csv))
    # every (dataset, seed, budget, E) is a cell with results and exactly one timing row
    for name, n_train in (("clf", 350), ("reg", 210)):
        d = df[df["dataset"] == name]
        assert sorted(d["budget"].unique()) == scaling_budgets(n_train, start=64)
        assert set(d["n_estimators"]) == {1, 2, 4}
        assert d.groupby(["budget", "n_estimators"])["seed"].nunique().eq(2).all()
    assert set(df["method"]) == {"mean", "mixture"}
    assert set(df.loc[df["budget"] == df["n_train"], "budget_key"]) == {"full"}
    assert len(timing) == df["cell"].nunique() and set(timing["cell"]) == set(df["cell"])
    for col in ("fit_seconds", "predict_seconds", "total_seconds"):
        assert (timing[col] >= 0).all()
    assert (timing["total_seconds"] >= timing["fit_seconds"] + timing["predict_seconds"] - 1e-6).all()
    assert (timing["device_name"] == "cpu").all()
    before = csv.read_bytes(), timing_path(csv).read_bytes()
    run_scaling(cfg)
    assert (csv.read_bytes(), timing_path(csv).read_bytes()) == before


def test_results_bit_identical_timing_separate(tmp_path):
    a = run_scaling(_cfg(tmp_path / "a", out_csv=str(tmp_path / "a" / "s.csv")))
    b = run_scaling(_cfg(tmp_path / "b", out_csv=str(tmp_path / "b" / "s.csv")))
    assert a.read_bytes() == b.read_bytes()
    assert not set(read_results(a)["metric"]) & {"fit_seconds", "predict_seconds", "peak_gpu_mb"}


def test_validate_flags_missing_timing(tmp_path):
    csv = run_scaling(_cfg(tmp_path))
    t = read_timing(timing_path(csv))
    t.iloc[1:].to_csv(timing_path(csv), index=False)
    assert any("no timing row" in p for p in cli.validate_csv(csv))
    timing_path(csv).unlink()
    assert "timing file missing" in cli.validate_csv(csv)


def test_shards_merge_with_timing(tmp_path):
    cfg = _cfg(tmp_path)
    for i in range(2):
        run_scaling(cfg, shard=(i, 2))
    out, problems = cli.merge(tmp_path / "tables" / "scaling_t.csv")
    assert problems == []
    assert len(read_timing(timing_path(out))) == read_results(out)["cell"].nunique()


def test_analyze_scaling_steps(tmp_path):
    csv = run_scaling(_cfg(tmp_path))
    res = cli.analyze(csv)
    s = res["summary"]
    assert {"mean", "se", "n_seeds", "total_seconds", "peak_gpu_mb"} <= set(s.columns)
    ctx = res["context_steps"]
    one = ctx[(ctx["dataset"] == "clf") & (ctx["metric"] == "log_loss") & (ctx["n_estimators"] == 1)]
    assert list(zip(one["budget_from"], one["budget_to"])) == [(64, 128), (128, 256), (256, 350)]
    est = res["estimator_steps"]
    assert set(zip(est["n_estimators_from"], est["n_estimators_to"])) == {(1, 2), (2, 4)}
    assert (est["n_seeds"] == 2).all()
    assert isinstance(pd.read_csv(csv.with_name("scaling_t.analysis.context_steps.csv")), pd.DataFrame)
