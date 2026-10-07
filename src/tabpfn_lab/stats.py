"""Paired, dataset-level statistics: seeds are averaged within a dataset first, then datasets are resampled."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .metrics import higher_is_better

ALIGN_COLUMNS = ["dataset", "split_seed", "seed", "budget", "n_estimators"]


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def sign_test(diffs) -> float:
    """Exact two-sided binomial sign test; zeros are dropped. Returns 1.0 when nothing is left."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 1.0
    return float(sps.binomtest(int((d > 0).sum()), len(d), 0.5).pvalue)


def wilcoxon_p(diffs) -> float:
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    try:
        return float(sps.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def win_rate(improvements) -> float:
    """Fraction of datasets where the method improves; ties count 0.5."""
    d = np.asarray(improvements, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d))


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def _flagged(flags, exclude_flags) -> pd.Series:
    flags = flags.fillna("").astype(str)
    mask = pd.Series(False, index=flags.index)
    for f in exclude_flags:
        mask |= flags.str.split(",").apply(lambda parts, f=f: f in parts)
    return mask


def paired_by_dataset(df, method, baseline, metric, exclude_flags=("saturated",)):
    """Per-dataset mean of (method - baseline), aligned on (dataset, split_seed, seed, budget, n_estimators).

    Returns (Series dataset -> mean raw difference, list of excluded datasets).
    """
    d = df[df["metric"] == metric]
    keys = [c for c in ALIGN_COLUMNS if c in d.columns]
    m = d[d["method"] == method]
    b = d[d["method"] == baseline]
    excluded: list[str] = []
    if exclude_flags and "flags" in m.columns:
        bad = _flagged(m["flags"], exclude_flags)
        excluded = sorted(m.loc[bad, "dataset"].unique().tolist())
        m = m[~bad]
    merged = m[keys + ["value"]].merge(b[keys + ["value"]], on=keys, suffixes=("_m", "_b"))
    merged["diff"] = merged["value_m"] - merged["value_b"]
    per_ds = merged.groupby("dataset")["diff"].mean()
    excluded = [e for e in excluded if e not in per_ds.index]
    return per_ds, excluded


def compare(df, method, baseline, metric, n_boot=2000, alpha=0.05, seed=0, exclude_flags=("saturated",)) -> dict:
    """Method vs baseline on one metric. `improvement` is oriented so that positive = better."""
    per_ds, excluded = paired_by_dataset(df, method, baseline, metric, exclude_flags)
    sign = 1.0 if higher_is_better(metric) else -1.0
    imp = sign * per_ds.to_numpy()
    lo, hi = bootstrap_ci(imp, n_boot=n_boot, alpha=alpha, seed=seed)
    p_sign = sign_test(imp)
    significant = bool(len(imp) >= 2 and np.isfinite(lo) and (lo > 0 or hi < 0) and p_sign < alpha)
    return {
        "method": method,
        "baseline": baseline,
        "metric": metric,
        "n_datasets": int(len(imp)),
        "mean_diff": float(per_ds.mean()) if len(per_ds) else float("nan"),
        "improvement": float(imp.mean()) if len(imp) else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "p_sign": p_sign,
        "p_wilcoxon": wilcoxon_p(imp),
        "win_rate": win_rate(imp),
        "significant": significant,
        "excluded": excluded,
    }


def compare_by_tier(df, method, baseline, metric, **kw) -> dict:
    """Per-tier comparisons, the pooled one, and a Kruskal-Wallis test for heterogeneity across tiers."""
    out = {"tiers": {}, "pooled": compare(df, method, baseline, metric, **kw)}
    groups = []
    sign = 1.0 if higher_is_better(metric) else -1.0
    for tier, sub in df.groupby("tier"):
        out["tiers"][tier] = compare(sub, method, baseline, metric, **kw)
        per_ds, _ = paired_by_dataset(sub, method, baseline, metric, kw.get("exclude_flags", ("saturated",)))
        if len(per_ds) >= 2:
            groups.append(sign * per_ds.to_numpy())
    if len(groups) >= 2:
        try:
            p = float(sps.kruskal(*groups).pvalue)
        except ValueError:
            p = 1.0
    else:
        p = float("nan")
    out["heterogeneity_p"] = p
    out["heterogeneous"] = bool(np.isfinite(p) and p < kw.get("alpha", 0.05))
    return out


def compare_family(df, family, baseline_for, alpha=0.05, **kw) -> pd.DataFrame:
    """Run `compare` for every (method, budget_key, metric) in the pre-registered family; add Holm-adjusted p."""
    rows = []
    for method, budget_key, metric in family:
        sub = df
        if budget_key is not None and "budget_key" in df.columns:
            sub = df[df["budget_key"] == budget_key]
        r = compare(sub, method, baseline_for(method), metric, alpha=alpha, **kw)
        r["budget_key"] = budget_key
        rows.append(r)
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm(res["p_sign"].to_numpy())
        res["significant_holm"] = (res["p_holm"] < alpha) & ((res["ci_low"] > 0) | (res["ci_high"] < 0))
    return res
