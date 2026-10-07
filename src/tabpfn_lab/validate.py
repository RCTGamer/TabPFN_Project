"""Results-schema validation. `validate_results` returns a list of human-readable problems (empty = clean)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import METRIC_DIRECTION, METRIC_RANGE

# section 3.5 schema; kind: "str", "int", "float"
SCHEMA = {
    "experiment": "str",
    "dataset": "str",
    "split": "str",
    "tier": "str",
    "task": "str",
    "seed": "int",
    "method": "str",
    "budget": "float",
    "ratio": "float",
    "n_train": "int",
    "metric": "str",
    "value": "float",
    "backend": "str",
    "role": "str",
    "flags": "str",
}
# extra columns the runner adds (split_seed: seeds of the split; n_estimators; budget_key: "f0.1"/"a500"/"lc"/"full"; cell id)
EXTRA_COLUMNS = {"split_seed": "int", "n_estimators": "int", "budget_key": "str", "cell": "str"}
COLUMNS = list(SCHEMA) + list(EXTRA_COLUMNS)
FULL_METHODS = {"full"}


def _key_columns(df):
    return [c for c in ["dataset", "split_seed", "seed", "method", "budget", "n_estimators", "metric"] if c in df.columns]


def _check_dtype(s: pd.Series, kind: str) -> bool:
    if kind == "str":
        return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
    if kind == "int":
        return pd.api.types.is_integer_dtype(s)
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _as_mapping(expected, datasets):
    if expected is None:
        return None
    if isinstance(expected, dict):
        return {d: list(expected.get(d, [])) for d in datasets}
    return {d: list(expected) for d in datasets}


def validate_results(
    df: pd.DataFrame,
    expected_methods=None,
    expected_datasets=None,
    expected_seeds=None,
    expected_budgets=None,
    failures: pd.DataFrame | None = None,
    required_baselines=None,
    min_random_budgets: int | None = None,
) -> list[str]:
    """Check schema, values and grid completeness.

    expected_methods: list, or {task: list}.  expected_datasets: list, or {dataset: task}.
    expected_budgets: list, or {dataset: list} (checked for non-full methods of the coreset experiment).
    required_baselines: methods that must exist at every (dataset, split_seed, seed, budget); `full` is
    checked once per (dataset, split_seed, seed).
    """
    problems: list[str] = []
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    for col, kind in {**SCHEMA, **EXTRA_COLUMNS}.items():
        if col in df.columns and len(df) and not _check_dtype(df[col], kind):
            problems.append(f"column {col!r} has dtype {df[col].dtype}, expected {kind}")
    if missing and not {"metric", "value", "dataset", "method"} <= set(df.columns):
        return problems

    if failures is not None and len(failures):
        for _, f in failures.iterrows():
            problems.append(f"failed cell {f.get('cell', '?')}: {f.get('error_type', '')}: {str(f.get('message', ''))[:200]}")

    unknown = sorted(set(df["metric"]) - set(METRIC_DIRECTION))
    if unknown:
        problems.append(f"unknown metric names: {unknown}")

    vals = pd.to_numeric(df["value"], errors="coerce")
    bad = ~np.isfinite(vals.to_numpy(dtype=float))
    if bad.any():
        problems.append(f"{int(bad.sum())} NaN/inf values (e.g. {df.loc[bad, ['dataset', 'method', 'metric']].head(3).to_dict('records')})")

    keys = _key_columns(df)
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate rows on {keys}")

    for metric, (lo, hi) in METRIC_RANGE.items():
        m = (df["metric"] == metric).to_numpy() & ~bad
        if not m.any():
            continue
        v = vals.to_numpy(dtype=float)[m]
        if (lo is not None and (v < lo - 1e-9).any()) or (hi is not None and (v > hi + 1e-9).any()):
            problems.append(f"out-of-range values for {metric}: min={v.min():.4g} max={v.max():.4g}, allowed [{lo}, {hi}]")

    if {"budget", "n_train", "method"} <= set(df.columns):
        over = df["budget"].notna() & (df["budget"] > df["n_train"]) & ~df["method"].isin(FULL_METHODS)
        if over.any():
            problems.append(f"{int(over.sum())} rows with budget > n_train for non-full methods: {sorted(df.loc[over, 'method'].unique())}")

    # ---- grid completeness
    if expected_datasets is not None:
        ds_task = expected_datasets if isinstance(expected_datasets, dict) else {d: None for d in expected_datasets}
        for d in ds_task:
            if d not in set(df["dataset"]):
                problems.append(f"dataset {d!r} missing from results")
    else:
        ds_task = {d: None for d in df["dataset"].unique()}
    present_tasks = df.groupby("dataset")["task"].first().to_dict() if "task" in df.columns else {}
    methods_by_task = expected_methods if isinstance(expected_methods, dict) else None
    budgets = _as_mapping(expected_budgets, list(ds_task)) if expected_budgets is not None else None
    for d, task in ds_task.items():
        sub = df[df["dataset"] == d]
        if not len(sub):
            continue
        task = task or present_tasks.get(d)
        if expected_methods is not None:
            methods = methods_by_task.get(task, []) if methods_by_task is not None else list(expected_methods)
            for m in methods:
                if m not in set(sub["method"]):
                    problems.append(f"method {m!r} missing for dataset {d!r}")
        if expected_seeds is not None:
            for s in expected_seeds:
                if s not in set(sub["seed"]):
                    problems.append(f"seed {s} missing for dataset {d!r}")
            if expected_methods is not None:
                methods = methods_by_task.get(task, []) if methods_by_task is not None else list(expected_methods)
                for m in methods:
                    have = set(sub.loc[sub["method"] == m, "seed"])
                    gone = [s for s in expected_seeds if s not in have]
                    if gone and m in set(sub["method"]):
                        problems.append(f"method {m!r} on {d!r} missing seeds {gone}")
        if budgets is not None:
            want = budgets.get(d, [])
            methods = (methods_by_task.get(task, []) if methods_by_task is not None else list(expected_methods or sub["method"].unique()))
            for m in methods:
                if m in FULL_METHODS:
                    continue
                have = set(sub.loc[sub["method"] == m, "budget"].dropna().astype(int))
                gone = [b for b in want if int(b) not in have]
                if gone:
                    problems.append(f"method {m!r} on {d!r} missing budgets {gone}")

    # ---- baselines present at every coreset cell
    if required_baselines:
        problems += _check_baselines(df, required_baselines)
    if min_random_budgets:
        for d, sub in df[df["method"] == "random"].groupby("dataset"):
            nb = sub["budget"].nunique()
            if nb < min_random_budgets:
                problems.append(f"random learning curve on {d!r} has {nb} budgets, need >= {min_random_budgets}")
        for d in set(df["dataset"]) - set(df.loc[df["method"] == "random", "dataset"]):
            problems.append(f"random learning curve missing for {d!r}")
    return problems


def _check_baselines(df, required) -> list[str]:
    problems = []
    seed_cols = [c for c in ["split_seed", "seed"] if c in df.columns]
    others = [m for m in required if m not in FULL_METHODS]
    coreset_rows = df[~df["method"].isin(FULL_METHODS) & df["budget"].notna()]
    for (d, *seeds), sub in df.groupby(["dataset"] + seed_cols):
        seeds = tuple(int(s) for s in seeds)
        task = sub["task"].iloc[0]
        if "full" in required and "full" not in set(sub["method"]):
            problems.append(f"baseline 'full' missing for {d!r} seeds {seeds}")
        cells = coreset_rows.loc[coreset_rows.index.intersection(sub.index)]
        grid_budgets = set(cells.loc[~cells["method"].isin(["random"]), "budget"].unique())
        for b in sorted(grid_budgets):
            at = set(cells.loc[cells["budget"] == b, "method"])
            for m in others:
                if m == "builtin_majority_downsample" and task != "classification":
                    continue
                if m not in at:
                    problems.append(f"baseline {m!r} missing for {d!r} seeds {seeds} budget {int(b)}")
    return problems
