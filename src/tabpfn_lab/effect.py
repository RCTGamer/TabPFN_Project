"""Effect sizes for experiment B (equivalent budget, gap closed), headroom flags, and power planning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.isotonic import IsotonicRegression

from .metrics import higher_is_better
from .stats import bootstrap_ci

SEED_COLUMNS = ["split_seed", "seed"]


# --------------------------------------------------------------------------- learning-curve inversion


def smooth_monotone(budgets, values, higher_better: bool) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic fit of the random-subset learning curve: non-decreasing quality with budget."""
    b = np.asarray(budgets, float)
    v = np.asarray(values, float)
    order = np.argsort(b)
    b, v = b[order], v[order]
    iso = IsotonicRegression(increasing=bool(higher_better))
    return b, iso.fit_transform(np.log(b), v)


def equivalent_budget(core_value, core_budget, rand_budgets, rand_values, higher_better: bool) -> tuple[float, str]:
    """(random budget matching the coreset's metric) / core_budget, from the monotone random curve.

    Interpolates linearly in log-budget. Outside the measured range returns (NaN, 'above_range' / 'below_range').
    """
    b, v = smooth_monotone(rand_budgets, rand_values, higher_better)
    q = v if higher_better else -v  # quality, non-decreasing in b
    c = core_value if higher_better else -core_value
    if c > q[-1]:
        return float("nan"), "above_range"
    if c < q[0]:
        return float("nan"), "below_range"
    # first point where quality reaches c
    j = int(np.argmax(q >= c))
    if j == 0 or q[j] == c:
        return float(b[j] / core_budget), ""
    lb0, lb1 = np.log(b[j - 1]), np.log(b[j])
    t = (c - q[j - 1]) / (q[j] - q[j - 1])
    return float(np.exp(lb0 + t * (lb1 - lb0)) / core_budget), ""


def gap_closed(m_random, m_core, m_full, higher_better: bool, noise: float = 0.0) -> tuple[float, str]:
    """(random - coreset) / (random - full) for lower-is-better (sign flipped otherwise). Not clipped.

    If |random - full| <= noise (no headroom) returns (NaN, 'saturated').
    """
    gap = m_random - m_full
    if not np.isfinite(gap) or abs(gap) <= max(noise, 1e-12):
        return float("nan"), "saturated"
    return float((m_random - m_core) / gap), ""


# --------------------------------------------------------------------------- headroom flags


def _add_flag(flags: pd.Series, mask, flag) -> pd.Series:
    flags = flags.fillna("").astype(str).copy()
    cur = flags[mask]
    flags[mask] = np.where(cur == "", flag, cur + "," + flag)
    return flags


def annotate_saturation(df: pd.DataFrame, se_mult: float = 1.0) -> pd.DataFrame:
    """Flag every (dataset, budget, metric) cell where random is within `se_mult` standard errors of full.

    The standard error is that of the paired (random - full) difference across seeds.
    """
    out = df.copy()
    if "flags" not in out.columns:
        out["flags"] = ""
    seed_cols = [c for c in SEED_COLUMNS if c in out.columns]
    full = out[out["method"] == "full"][["dataset", "metric", "value"] + seed_cols]
    rand = out[out["method"] == "random"]
    merged = rand.merge(full, on=["dataset", "metric"] + seed_cols, suffixes=("", "_full"))
    merged["d"] = merged["value"] - merged["value_full"]
    for (d, b, metric), g in merged.groupby(["dataset", "budget", "metric"]):
        if len(g) < 2:
            continue
        se = g["d"].std(ddof=1) / np.sqrt(len(g))
        if abs(g["d"].mean()) < se_mult * se or g["d"].abs().max() == 0:
            mask = (out["dataset"] == d) & (out["budget"] == b) & (out["metric"] == metric) & (out["method"] != "full")
            out["flags"] = _add_flag(out["flags"], mask, "saturated")
    return out


# --------------------------------------------------------------------------- effect tables


def effect_sizes(df: pd.DataFrame, method: str, metric: str, baseline: str = "random") -> pd.DataFrame:
    """Per (dataset, budget): seed-averaged coreset/random/full metrics, gap_closed, equivalent_budget_multiplier."""
    hb = higher_is_better(metric)
    d = df[df["metric"] == metric]
    seed_cols = [c for c in SEED_COLUMNS if c in d.columns]
    rows = []
    for ds, g in d.groupby("dataset"):
        full = g[g["method"] == "full"]
        rand = g[g["method"] == baseline]
        core = g[g["method"] == method]
        if full.empty or rand.empty or core.empty:
            continue
        m_full = full["value"].mean()
        curve = rand.groupby("budget")["value"].mean()
        for b, cg in core.groupby("budget"):
            rg = rand[rand["budget"] == b]
            if rg.empty:
                continue
            paired = rg.merge(full[seed_cols + ["value"]], on=seed_cols, suffixes=("", "_full"))
            diff = paired["value"] - paired["value_full"]
            noise = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) >= 2 else 0.0
            m_core, m_rand = cg["value"].mean(), rg["value"].mean()
            gc, gflag = gap_closed(m_rand, m_core, m_full, hb, noise=noise)
            mult, eflag = equivalent_budget(m_core, b, curve.index.to_numpy(), curve.to_numpy(), hb) if len(curve) >= 2 else (float("nan"), "no_curve")
            rows.append(
                {
                    "dataset": ds,
                    "tier": g["tier"].iloc[0] if "tier" in g else "",
                    "split": g["split"].iloc[0] if "split" in g else "",
                    "budget": b,
                    "budget_key": cg["budget_key"].iloc[0] if "budget_key" in cg else "",
                    "ratio": cg["ratio"].iloc[0] if "ratio" in cg else b / g["n_train"].iloc[0],
                    "method": method,
                    "metric": metric,
                    "m_core": m_core,
                    "m_random": m_rand,
                    "m_full": m_full,
                    "gap_closed": gc,
                    "equivalent_budget_multiplier": mult,
                    "flags": ",".join(f for f in (gflag, eflag) if f),
                }
            )
    return pd.DataFrame(rows)


def summarize_effect(effects: pd.DataFrame, column: str, n_boot=2000, seed=0) -> dict:
    """Dataset-level mean and bootstrap CI of an effect column (datasets weighted equally; NaN cells skipped)."""
    per_ds = effects.groupby("dataset")[column].mean().dropna()
    lo, hi = bootstrap_ci(per_ds.to_numpy(), n_boot=n_boot, seed=seed)
    return {"column": column, "n_datasets": int(len(per_ds)), "mean": float(per_ds.mean()) if len(per_ds) else float("nan"), "ci_low": lo, "ci_high": hi}


def ratio_curve(effects: pd.DataFrame, column: str = "gap_closed") -> pd.DataFrame:
    """One row per reduction ratio (sorted): dataset-weighted mean of `column`, number of datasets, NaN cells skipped."""
    if effects.empty:
        return pd.DataFrame(columns=["ratio", column, "n_datasets"])
    e = effects.copy()
    e["ratio"] = e["ratio"].astype(float).round(6)
    per = e.groupby(["ratio", "dataset"])[column].mean().reset_index()
    out = per.groupby("ratio")[column].agg(["mean", lambda s: int(s.notna().sum())]).reset_index()
    out.columns = ["ratio", column, "n_datasets"]
    return out.sort_values("ratio").reset_index(drop=True)


# --------------------------------------------------------------------------- power / planning


def estimate_noise(df: pd.DataFrame, method: str, baseline: str, metric: str, relative: bool = True) -> dict:
    """Variance components of paired differences: sd across datasets (of per-dataset means) and within-dataset sd across seeds."""
    d = df[df["metric"] == metric]
    keys = [c for c in ["dataset", "split_seed", "seed", "budget", "n_estimators"] if c in d.columns]
    m = d[d["method"] == method][keys + ["value"]]
    b = d[d["method"] == baseline][keys + ["value"]]
    j = m.merge(b, on=keys, suffixes=("_m", "_b"))
    j["diff"] = j["value_m"] - j["value_b"]
    if relative:
        j["diff"] = j["diff"] / j["value_b"].abs().clip(lower=1e-12)
    per_ds = j.groupby("dataset")["diff"]
    sd_seed = float(np.sqrt(per_ds.var(ddof=1).fillna(0).mean())) if len(j) else float("nan")
    means = per_ds.mean()
    sd_dataset = float(means.std(ddof=1)) if len(means) >= 2 else float("nan")
    return {"sd_dataset": sd_dataset, "sd_seed": sd_seed, "n_datasets": int(len(means)), "seeds_per_dataset": float(per_ds.size().mean()) if len(j) else 0.0}


def min_detectable_effect_approx(n_datasets, n_seeds, sd_dataset, sd_seed, alpha=0.05, power=0.8) -> float:
    """Closed-form normal approximation (sign-test efficiency sqrt(pi/2) folded in). Optimistic for small D;
    use `min_detectable_effect` for planning."""
    sd = np.sqrt(sd_dataset**2 + sd_seed**2 / max(n_seeds, 1))
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return float(z * sd / np.sqrt(max(n_datasets, 1)) * np.sqrt(np.pi / 2))


def min_detectable_effect(n_datasets, n_seeds, sd_dataset, sd_seed, alpha=0.05, power=0.8, n_sim=400, seed=0) -> float:
    """Smallest true effect the decision rule detects with probability >= `power`, by bisection on
    `simulate_power` with common random numbers. Returns inf when no effect can reach the power
    (e.g. with <= 5 datasets the exact sign test can never give p < 0.05)."""
    sd = np.sqrt(sd_dataset**2 + sd_seed**2 / max(n_seeds, 1))
    if sd == 0:
        return 0.0
    pw = lambda e: simulate_power(n_datasets, n_seeds, e, sd_dataset, sd_seed, alpha, n_sim=n_sim, n_boot=300, seed=seed)  # noqa: E731
    lo, hi = 0.0, 4 * sd / np.sqrt(max(n_datasets, 1))
    for _ in range(6):
        if pw(hi) >= power:
            break
        lo, hi = hi, 2 * hi
    else:
        return float("inf")
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pw(mid) < power else (lo, mid)
    return float(hi)


def simulate_power(n_datasets, n_seeds, effect, sd_dataset, sd_seed, alpha=0.05, n_sim=300, n_boot=400, seed=0) -> float:
    """Fraction of simulated experiments the decision rule (bootstrap CI over datasets excludes 0 and exact
    sign-test p < alpha) calls significant. Vectorised; draws do not depend on `effect` (common random numbers)."""
    rng = np.random.default_rng(seed)
    D = int(n_datasets)
    per_ds = effect + rng.normal(0, sd_dataset, (n_sim, D)) + rng.normal(0, sd_seed, (n_sim, D, max(int(n_seeds), 1))).mean(-1)
    idx = np.random.default_rng([seed, 1]).integers(0, D, (n_boot, D))
    means = per_ds[:, idx].mean(-1)  # (n_sim, n_boot)
    lo = np.quantile(means, alpha / 2, axis=1)
    hi = np.quantile(means, 1 - alpha / 2, axis=1)
    k = (per_ds > 0).sum(1)
    n = (per_ds != 0).sum(1)
    p_sign = np.where(n > 0, np.minimum(1.0, 2 * sps.binom.cdf(np.minimum(k, n - k), np.maximum(n, 1), 0.5)), 1.0)
    return float(np.mean(((lo > 0) | (hi < 0)) & (p_sign < alpha)))


def plan(n_datasets, n_seeds, sd_dataset, sd_seed, mde_target, alpha=0.05, n_sim=300, seed=0) -> dict:
    mde = min_detectable_effect(n_datasets, n_seeds, sd_dataset, sd_seed, alpha, n_sim=n_sim, seed=seed)
    pw = simulate_power(n_datasets, n_seeds, mde_target, sd_dataset, sd_seed, alpha, n_sim=n_sim, seed=seed)
    return {"n_datasets": n_datasets, "n_seeds": n_seeds, "sd_dataset": sd_dataset, "sd_seed": sd_seed, "min_detectable_effect": mde, "target_effect": mde_target, "power": pw}
