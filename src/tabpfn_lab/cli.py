"""Command line: python -m tabpfn_lab.cli {aggregation|coreset|scaling|analyze|merge|plan|validate} --config ...

Designed for tmux/nohup on a GPU server: no prompts, line-buffered logs in results/logs/, resumable, shardable.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import effect, stats
from .experiments import (
    CANARY,
    REFERENCE_METHODS_B,
    dataset_specs,
    failures_path,
    meta_path,
    normalize_config,
    read_results,
    read_timing,
    run_aggregation,
    run_coreset,
    run_scaling,
    timing_path,
    validate_timing,
)
from .metrics import CLF_METRICS, REG_METRICS, higher_is_better
from .validate import validate_results


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_shard(s: str | None):
    if not s:
        return None
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"bad shard {s}")
    return i, n


# --------------------------------------------------------------------------- validation and merge


def validate_csv(csv: Path) -> list[str]:
    df = read_results(csv)
    mp = meta_path(csv)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    exp = meta.get("expected", {})
    fp = failures_path(csv)
    failures = None
    if fp.exists():
        failures = pd.read_csv(fp)
        failures = failures[~failures["cell"].isin(set(df["cell"]))].drop_duplicates("cell", keep="last")
    problems = validate_results(
        df,
        expected_methods=exp.get("methods"),
        expected_datasets=exp.get("datasets"),
        expected_seeds=exp.get("seeds"),
        expected_budgets=exp.get("budgets"),
        failures=failures,
        required_baselines=exp.get("required_baselines"),
        min_random_budgets=6 if meta.get("experiment") == "coreset" and "random" in meta.get("config", {}).get("coreset_methods", []) else None,
    )
    if meta.get("experiment") == "scaling":
        tp = timing_path(csv)
        problems += validate_timing(df, read_timing(tp) if tp.exists() else None)
    return problems


def merge(out_csv: Path) -> tuple[Path, list[str]]:
    """Concatenate `<name>.shard*.csv` (and failures / meta) into `<name>.csv`, then validate."""
    out_csv = Path(out_csv)
    pattern = str(out_csv.with_name(f"{out_csv.stem}.shard*{out_csv.suffix}"))
    shards = sorted(p for p in glob.glob(pattern) if ".failures" not in p and ".timing" not in p and not p.endswith(".meta.json"))
    if not shards:
        raise FileNotFoundError(f"no shard files match {pattern}")
    df = pd.concat([read_results(p) for p in shards], ignore_index=True)
    df = df.sort_values(["cell", "method", "metric", "n_estimators"], kind="stable").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    fails = [pd.read_csv(failures_path(Path(p))) for p in shards if failures_path(Path(p)).exists()]
    if fails:
        pd.concat(fails, ignore_index=True).to_csv(failures_path(out_csv), index=False)
    timings = [read_timing(timing_path(Path(p))) for p in shards if timing_path(Path(p)).exists()]
    if timings:
        pd.concat(timings, ignore_index=True).sort_values("cell", kind="stable").to_csv(timing_path(out_csv), index=False)
    metas = [json.loads(meta_path(Path(p)).read_text()) for p in shards if meta_path(Path(p)).exists()]
    if metas:
        m = dict(metas[0])
        m["shard"] = None
        m["merged_from"] = shards
        m["split_hashes"] = {k: v for mm in metas for k, v in mm.get("split_hashes", {}).items()}
        m["backend_meta"] = next((mm["backend_meta"] for mm in metas if mm.get("backend_meta")), {})
        if "timing_csv" in m:
            m["timing_csv"] = str(timing_path(out_csv))
        meta_path(out_csv).write_text(json.dumps(m, indent=2, sort_keys=True))
    return out_csv, validate_csv(out_csv)


# --------------------------------------------------------------------------- analysis


def _fmt(r: dict) -> str:
    star = "*" if r.get("significant_holm", r.get("significant")) else " "
    p = r.get("p_holm", r["p_sign"])
    return (
        f"{r['method']:<28} {r['metric']:<9} n={r['n_datasets']:<3} diff={r['mean_diff']:+.4f} "
        f"imp={r['improvement']:+.4f} CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] p={p:.3g} win={r['win_rate']:.2f} {star}"
    )


def analyze_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, baseline, metrics in (("classification", "mean", CLF_METRICS), ("regression", "mixture", REG_METRICS)):
        d = df[df["task"] == task]
        if d.empty:
            continue
        for E in sorted(d["n_estimators"].unique()):
            dE = d[d["n_estimators"] == E]
            for m in sorted(set(dE["method"]) - {baseline}):
                for metric in metrics:
                    r = stats.compare(dE, m, baseline, metric)
                    r.update(n_estimators=int(E), task=task)
                    rows.append(r)
    return pd.DataFrame(rows)


def analyze_coreset(df: pd.DataFrame, meta: dict) -> dict[str, pd.DataFrame]:
    df = effect.annotate_saturation(df)
    confirm = df[df["split"] == "confirm"]
    family = [tuple(f) for f in meta.get("family", [])]
    out = {}
    baseline = meta.get("config", {}).get("baseline", "random")
    out["family"] = stats.compare_family(confirm, family, baseline_for=lambda m: baseline) if family else pd.DataFrame()
    effects = []
    for task, metric in (("classification", "log_loss"), ("regression", "rmse")):
        d = confirm[confirm["task"] == task]
        # every method incl. tabpfn's builtin_* samplers, measured against random and full data
        for m in sorted(set(d["method"]) - {"full", "random", CANARY}):
            e = effect.effect_sizes(d, m, metric)
            if not e.empty:
                effects.append(e)
    eff = pd.concat(effects, ignore_index=True) if effects else pd.DataFrame()
    out["effects"] = eff
    summary = []
    if not eff.empty:
        for (m, metric), g in eff.groupby(["method", "metric"]):
            for col in ("gap_closed", "equivalent_budget_multiplier"):
                s = effect.summarize_effect(g[~g["flags"].str.contains("saturated")], col)
                s.update(method=m, metric=metric)
                summary.append(s)
        curves = []
        for (m, metric, tier), g in eff.groupby(["method", "metric", "tier"]):
            c = effect.ratio_curve(g[~g["flags"].str.contains("saturated")])
            c["method"], c["metric"], c["tier"] = m, metric, tier
            curves.append(c)
        out["ratio_curves"] = pd.concat(curves, ignore_index=True)
    out["effect_summary"] = pd.DataFrame(summary)
    out["saturated_cells"] = df[df["flags"].str.contains("saturated")][["dataset", "budget", "metric"]].drop_duplicates()
    return out


def _step_table(d: pd.DataFrame, along: str, fixed: list[str]) -> pd.DataFrame:
    """Paired effect of moving one step up `along` (budget or n_estimators) with `fixed` held, matched on seed.

    improvement > 0 means the larger setting is better; p is a two-sided paired t-test over seeds, Holm-adjusted
    within the returned table.
    """
    from scipy.stats import ttest_rel

    rows = []
    for key, g in d.groupby(fixed):
        levels = sorted(g[along].unique())
        wide = g.pivot_table(index="seed", columns=along, values="value", aggfunc="first")
        sign = 1.0 if higher_is_better(g["metric"].iloc[0]) else -1.0
        for lo, hi in itertools.pairwise(levels):
            pair = wide[[lo, hi]].dropna()
            imp = sign * (pair[hi] - pair[lo])
            n = len(imp)
            p = float(ttest_rel(pair[hi], pair[lo]).pvalue) if n >= 2 and imp.std() > 0 else float("nan")
            rows.append({**dict(zip(fixed, key)), f"{along}_from": lo, f"{along}_to": hi, "n_seeds": n,
                         "value_from": float(pair[lo].mean()) if n else float("nan"), "value_to": float(pair[hi].mean()) if n else float("nan"),
                         "improvement": float(imp.mean()) if n else float("nan"),
                         "se": float(imp.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan"), "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        ok = out["p"].notna()
        out["p_holm"] = np.nan
        out.loc[ok, "p_holm"] = stats.holm(out.loc[ok, "p"].to_numpy())
        out["significant_holm"] = out["p_holm"] < 0.05
    return out


def analyze_scaling(df: pd.DataFrame, timing: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Mean metric and cost per (dataset, budget, n_estimators), plus paired one-step effects of doubling the
    context at fixed E and of the next estimator count at fixed context."""
    keys = ["dataset", "task", "method", "metric", "budget", "n_estimators"]
    g = df.groupby(keys)["value"]
    summary = g.mean().rename("mean").to_frame().join(g.std().rename("sd")).join(g.count().rename("n_seeds")).reset_index()
    summary["se"] = summary["sd"] / np.sqrt(summary["n_seeds"])
    if timing is not None and len(timing):
        cost = timing.groupby(["dataset", "budget", "n_estimators"])[["fit_seconds", "predict_seconds", "total_seconds", "peak_gpu_mb"]].mean()
        summary = summary.merge(cost.reset_index(), on=["dataset", "budget", "n_estimators"], how="left")
    ctx_steps = _step_table(df, "budget", ["dataset", "task", "method", "metric", "n_estimators"])
    est_steps = _step_table(df, "n_estimators", ["dataset", "task", "method", "metric", "budget"])
    return {"summary": summary, "context_steps": ctx_steps, "estimator_steps": est_steps}


def analyze_coreset_dev(df: pd.DataFrame, baseline: str, timing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Exploratory table for a coreset run without confirm datasets: per dataset, budget, method and metric, the mean
    over (split_seed, seed) and the paired difference to `baseline` (improvement > 0 is better), Holm-adjusted."""
    from scipy.stats import ttest_rel

    rows = []
    for (d, task, b, metric), g in df.groupby(["dataset", "task", "budget", "metric"]):
        wide = g.pivot_table(index=["split_seed", "seed"], columns="method", values="value", aggfunc="first")
        sign = 1.0 if higher_is_better(metric) else -1.0
        for m in wide.columns:
            r = {"dataset": d, "task": task, "budget": b, "metric": metric, "method": m, "n": int(wide[m].notna().sum()),
                 "mean": float(wide[m].mean()), "se": float(wide[m].std(ddof=1) / np.sqrt(wide[m].notna().sum()))}
            r.update(improvement=np.nan, imp_se=np.nan, p=np.nan)
            if baseline in wide.columns and m != baseline:
                pair = wide[[m, baseline]].dropna()
                imp = sign * (pair[m] - pair[baseline])
                if len(imp) >= 2:
                    r.update(improvement=float(imp.mean()), imp_se=float(imp.std(ddof=1) / np.sqrt(len(imp))))
                    if imp.std() > 0:
                        r["p"] = float(ttest_rel(pair[m], pair[baseline]).pvalue)
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        ok = out["p"].notna()
        out["p_holm"] = np.nan
        out.loc[ok, "p_holm"] = stats.holm(out.loc[ok, "p"].to_numpy())
        out["significant_holm"] = out["p_holm"] < 0.05
        if timing is not None and "method" in timing.columns:
            cost = timing.groupby(["dataset", "method"])[["total_seconds", "selection_seconds", "n_fits"]].mean().reset_index()
            # extra rows (`+gate`, `_corrected`, `+aggregator`) share the cost of the cell that produced them
            out["cell_method"] = out["method"].str.replace(r"(\+.*|_corrected)$", "", regex=True)
            out = out.merge(cost.rename(columns={"method": "cell_method"}), on=["dataset", "cell_method"], how="left").drop(columns="cell_method")
    return out


def analyze(csv: Path, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    csv = Path(csv)
    df = read_results(csv)
    mp = meta_path(csv)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    out_dir = Path(out_dir or csv.parent)
    if (df["experiment"] == "scaling").any():
        tp = timing_path(csv)
        res = analyze_scaling(df, read_timing(tp) if tp.exists() else None)
        primary = res["summary"]["metric"].isin(["log_loss", "rmse"])
        cols = [c for c in ["dataset", "method", "metric", "budget", "n_estimators", "mean", "se", "n_seeds", "total_seconds", "peak_gpu_mb"] if c in res["summary"]]
        print("== scaling: mean over seeds (primary metrics; full tables in the .analysis.*.csv files) ==")
        print(res["summary"].loc[primary, cols].to_string(index=False))
        for k, title in (("context_steps", "doubling the context at fixed E"), ("estimator_steps", "next estimator count at fixed context")):
            t = res[k]
            if len(t):
                print(f"\n== {title}: improvement > 0 is better, Holm-adjusted p ==")
                print(t[t["metric"].isin(["log_loss", "rmse"])].to_string(index=False))
    elif (df["experiment"] == "coreset").any() and not (df["split"] == "confirm").any():
        baseline = meta.get("config", {}).get("baseline", "random")
        tp = timing_path(csv)
        res = {"dev": analyze_coreset_dev(df, baseline, read_timing(tp) if tp.exists() else None)}
        t = res["dev"]
        cols = [c for c in ["dataset", "method", "metric", "mean", "se", "improvement", "imp_se", "p_holm", "significant_holm", "total_seconds"] if c in t]
        print(f"== coreset, dev datasets only (exploratory): paired vs {baseline}, improvement > 0 is better, Holm-adjusted ==")
        print(t.loc[t["metric"].isin(["log_loss", "rmse"]), cols].sort_values(["dataset", "improvement"], ascending=[True, False]).to_string(index=False))
    elif (df["experiment"] == "coreset").any():
        res = analyze_coreset(df, meta)
        print(f"== Experiment B (coreset) vs {meta.get('config', {}).get('baseline', 'random')}, confirm datasets, Holm over the pre-registered family ==")
        fam = res["family"]
        for _, r in fam.iterrows() if len(fam) else []:
            print(f"[{r['budget_key']:>6}] " + _fmt(r.to_dict()))
        if len(res["effect_summary"]):
            print("\n== effect sizes (dataset-weighted, saturated cells excluded) ==")
            print(res["effect_summary"].to_string(index=False))
        if len(res["saturated_cells"]):
            print(f"\n{len(res['saturated_cells'])} saturated (dataset, budget, metric) cells listed but excluded from claims")
    else:
        res = {"comparisons": analyze_aggregation(df)}
        print("== Experiment A (aggregation) vs mean / mixture ==")
        for _, r in res["comparisons"].iterrows():
            print(f"[E={r['n_estimators']}] " + _fmt(r.to_dict()))
    for k, v in res.items():
        if isinstance(v, pd.DataFrame) and len(v):
            v.to_csv(out_dir / f"{csv.stem}.analysis.{k}.csv", index=False)
    return res


# --------------------------------------------------------------------------- planning


def plan_from_config(cfg: dict, pilot: str | None = None, n_sim=300) -> dict:
    cfg = normalize_config(cfg, "coreset")
    specs = dataset_specs(cfg)
    D = sum(1 for d in specs if d["split"] == "confirm")
    S = len(cfg["split_seeds"]) * len(cfg["selection_seeds"])
    target = float(cfg.get("min_detectable_effect", 0.01))
    alpha = float(cfg.get("alpha", 0.05))
    if pilot:
        df = read_results(pilot)
        sds = []
        for task, metric in (("classification", "log_loss"), ("regression", "rmse")):
            d = df[df["task"] == task]
            for m in sorted(set(d["method"]) - REFERENCE_METHODS_B):
                n = effect.estimate_noise(d, m, "random", metric)
                if np.isfinite(n["sd_dataset"]):
                    sds.append(n)
        if not sds:
            raise ValueError("pilot has no (method, random) pairs to estimate noise from")
        sd_dataset = max(s["sd_dataset"] for s in sds)
        sd_seed = max(s["sd_seed"] for s in sds)
    else:
        p = cfg.get("planning", {})
        sd_dataset, sd_seed = float(p.get("sd_dataset", 0.02)), float(p.get("sd_seed", 0.02))
    return effect.plan(D, S, sd_dataset, sd_seed, target, alpha=alpha, n_sim=n_sim)


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tabpfn_lab.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("aggregation", "coreset", "scaling"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--shard", type=parse_shard, default=None, help="i/N: run every N-th cell starting at i")
    p = sub.add_parser("analyze")
    p.add_argument("--config")
    p.add_argument("--csv")
    p.add_argument("--experiment", choices=["aggregation", "coreset", "scaling"], default=None)
    p = sub.add_parser("merge")
    p.add_argument("--name", help="config name; merges results/tables/<name>.shard*.csv")
    p.add_argument("--config")
    p.add_argument("--experiment", choices=["aggregation", "coreset", "scaling"], default=None)
    p.add_argument("--out-csv")
    p = sub.add_parser("validate")
    p.add_argument("--csv", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", required=True)
    p.add_argument("--pilot")
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-sim", type=int, default=300)
    args = ap.parse_args(argv)

    if args.cmd == "aggregation":
        csv = run_aggregation(load_config(args.config), shard=args.shard)
        problems = validate_csv(csv) if args.shard is None else []
        _report(problems)
        return 0
    if args.cmd == "coreset":
        csv = run_coreset(load_config(args.config), shard=args.shard)
        problems = validate_csv(csv) if args.shard is None else []
        _report(problems)
        return 0
    if args.cmd == "scaling":
        csv = run_scaling(load_config(args.config), shard=args.shard)
        problems = validate_csv(csv) if args.shard is None else []
        _report(problems)
        return 0
    if args.cmd == "analyze":
        csv = _csv_from_args(args)
        analyze(csv)
        return 0
    if args.cmd == "merge":
        csv = _csv_from_args(args)
        out, problems = merge(csv)
        print(f"merged -> {out}")
        _report(problems)
        return 1 if problems else 0
    if args.cmd == "validate":
        problems = validate_csv(Path(args.csv))
        _report(problems)
        return 1 if problems else 0
    if args.cmd == "plan":
        res = plan_from_config(load_config(args.config), args.pilot, n_sim=args.n_sim)
        print(json.dumps(res, indent=2))
        if res["power"] < 0.8:
            msg = f"power {res['power']:.2f} < 0.8 for min_detectable_effect={res['target_effect']} (MDE at 80% power: {res['min_detectable_effect']:.4f})"
            if not args.force:
                print("REFUSED: " + msg + "; add datasets/seeds or pass --force", file=sys.stderr)
                return 2
            print("WARNING (forced): " + msg, file=sys.stderr)
        return 0
    return 1


def _csv_from_args(args) -> Path:
    if getattr(args, "csv", None):
        return Path(args.csv)
    if getattr(args, "out_csv", None):
        return Path(args.out_csv)
    if getattr(args, "config", None):
        cfg = load_config(args.config)
        exp = args.experiment or ("coreset" if "coreset_methods" in cfg else "scaling" if "budget_start" in cfg else "aggregation")
        return Path(normalize_config(cfg, exp)["out_csv"])
    if getattr(args, "name", None):
        hits = [p for p in glob.glob(f"results/tables/{args.name}*.shard0.csv")]
        if len(hits) == 1:
            return Path(hits[0].replace(".shard0", ""))
        return Path(f"results/tables/{args.name}.csv")
    raise SystemExit("need --csv, --config or --name")


def _report(problems: list[str]) -> None:
    if problems:
        print(f"validate_results: {len(problems)} problem(s)", file=sys.stderr)
        for p in problems[:50]:
            print("  - " + p, file=sys.stderr)
    else:
        print("validate_results: clean")


if __name__ == "__main__":
    sys.exit(main())
