import json

import pytest

from tabpfn_lab import cli, experiments
from tabpfn_lab.experiments import TuningOnConfirmError, read_results, run_coreset, tune_coreset, validate_config
from tabpfn_lab.validate import validate_results

from conftest import SpyBackend

D = lambda name, seed: {"name": name, "source": "synthetic", "task": "classification", "n": 300, "imbalance": [0.8, 0.12, 0.08], "noise": 0.02, "data_seed": seed}  # noqa: E731


def test_dev_confirm_disjoint():
    with pytest.raises(ValueError, match="overlap"):
        validate_config({"datasets": {"dev": [D("a", 1)], "confirm": [D("a", 1), D("b", 2)]}})
    validate_config({"datasets": {"dev": [D("a", 1)], "confirm": [D("b", 2)]}})


def test_tuning_only_on_dev(smoke, monkeypatch):
    cfg = smoke(datasets={"dev": [D("dev1", 1)], "confirm": [D("conf1", 2)]}, budget_fractions=[0.3], selection_seeds=[0], n_estimators=1)
    seen = []
    real = experiments.evaluate_coreset

    def spy(cfg_, ctx, split, *a, **kw):
        seen.append(split.name)
        return real(cfg_, ctx, split, *a, **kw)

    monkeypatch.setattr(experiments, "evaluate_coreset", spy)
    best, summary = tune_coreset(cfg, "uncertainty_mix", [{"frac_hard": 0.2, "pilot": 50}, {"frac_hard": 0.8, "pilot": 50}], backend=SpyBackend())
    assert best["frac_hard"] in (0.2, 0.8)
    assert seen and set(seen) == {"dev1"}
    with pytest.raises(TuningOnConfirmError):
        tune_coreset(cfg, "uncertainty_mix", [{"frac_hard": 0.5}], datasets=["conf1"], backend=SpyBackend())


def _cfg(smoke, **kw):
    base = dict(
        datasets={"dev": [D("dev1", 1)], "confirm": [D("c1", 2), D("c2", 3)]},
        coreset_methods=["random", "stratified", "builtin_subsample", "builtin_majority_downsample"],
        budget_fractions=[0.5, 0.3],
        selection_seeds=[0, 1],
        n_estimators=2,
    )
    base.update(kw)
    return smoke(**base)


def test_family_defined_before_running(smoke, monkeypatch, capsys):
    cfg = _cfg(smoke)
    written = {}
    real_cell = experiments._coreset_cell

    def first_cell(cfg_, ctx, cell):
        if not written:
            from pathlib import Path

            p = Path(cfg_["out_csv"])
            written.update(json.loads(p.with_name(p.stem + ".meta.json").read_text()))
        return real_cell(cfg_, ctx, cell)

    monkeypatch.setattr(experiments, "_coreset_cell", first_cell)
    csv = run_coreset(cfg, backend=SpyBackend())
    fam = [tuple(f) for f in written["family"]]
    assert ("stratified", "f0.5", "log_loss") in fam and all(m not in experiments.REFERENCE_METHODS_B for m, _, _ in fam)
    res = cli.analyze(csv)
    assert [tuple(r) for r in res["family"][["method", "budget_key", "metric"]].itertuples(index=False)] == fam


def test_plan_blocks_underpowered_runs(tmp_path, smoke):
    import yaml

    cfg = _cfg(smoke, min_detectable_effect=0.001, planning={"sd_dataset": 0.05, "sd_seed": 0.05})
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert cli.main(["plan", "--config", str(p), "--n-sim", "100"]) == 2
    assert cli.main(["plan", "--config", str(p), "--n-sim", "100", "--force"]) == 0
    many = dict(cfg, datasets={"dev": [], "confirm": [D(f"c{i}", i) for i in range(40)]}, selection_seeds=10, min_detectable_effect=0.05, planning={"sd_dataset": 0.01, "sd_seed": 0.01})
    p.write_text(yaml.safe_dump(many))
    assert cli.main(["plan", "--config", str(p), "--n-sim", "100"]) == 0


def test_baselines_present_at_every_cell(smoke):
    csv = run_coreset(_cfg(smoke), backend=SpyBackend())
    df = read_results(csv)
    req = ["full", "random", "builtin_subsample", "builtin_majority_downsample"]
    assert validate_results(df, required_baselines=req) == []
    for m in req:
        problems = validate_results(df[df["method"] != m], required_baselines=req)
        assert any(f"'{m}' missing" in p for p in problems), m


def test_random_learning_curve_present(smoke):
    csv = run_coreset(_cfg(smoke), backend=SpyBackend())
    df = read_results(csv)
    per = df[df["method"] == "random"].groupby("dataset")["budget"].nunique()
    assert (per >= 6).all()
    assert validate_results(df, min_random_budgets=6) == []
    no_lc = df[~((df["method"] == "random") & (df["budget_key"] == "lc"))]
    assert validate_results(no_lc, min_random_budgets=6)
