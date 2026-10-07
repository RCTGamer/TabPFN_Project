import numpy as np
import pandas as pd
import pytest

from tabpfn_lab import effect as E
from tabpfn_lab import stats as S
from tabpfn_lab.sizes import LearningCurve

CURVE = LearningCurve(m_inf=0.3, a=2.0, alpha=0.5)  # log-loss-like, lower is better
BUDGETS = np.geomspace(50, 5000, 8)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_equivalent_budget_recovers_known_multiplier(k):
    est = []
    for seed in range(20):
        rand = CURVE.sample(BUDGETS, noise=0.002, seed=seed)
        b = 400.0
        core = CURVE.metric(k * b) + np.random.default_rng(seed + 100).normal(0, 0.002)
        mult, flag = E.equivalent_budget(core, b, BUDGETS, rand, higher_better=False)
        assert flag == ""
        est.append(mult)
    assert np.median(est) == pytest.approx(k, rel=0.15)


def test_equivalent_budget_no_silent_extrapolation():
    rand = CURVE.metric(BUDGETS)
    m, f = E.equivalent_budget(CURVE.metric(1e6), 100, BUDGETS, rand, higher_better=False)
    assert np.isnan(m) and f == "above_range"
    m, f = E.equivalent_budget(CURVE.metric(1.0), 100, BUDGETS, rand, higher_better=False)
    assert np.isnan(m) and f == "below_range"
    # higher-is-better mirror
    m, f = E.equivalent_budget(-CURVE.metric(1e6), 100, BUDGETS, -rand, higher_better=True)
    assert np.isnan(m) and f == "above_range"


def test_learning_curve_is_smoothed_monotone():
    mults = []
    for seed in range(10):
        noisy = CURVE.sample(BUDGETS, noise=0.02, seed=seed)
        b, v = E.smooth_monotone(BUDGETS, noisy, higher_better=False)
        assert (np.diff(v) <= 1e-12).all()  # lower-is-better metric never gets worse with more data
        m, _ = E.equivalent_budget(CURVE.metric(800), 400, BUDGETS, noisy, higher_better=False)
        mults.append(m)
    mults = np.array(mults)
    assert np.nanstd(np.log(mults[np.isfinite(mults)])) < 0.6


def test_gap_closed_formula():
    # lower is better: random 0.5, full 0.3, coreset 0.4 -> (0.5-0.4)/(0.5-0.3) = 0.5
    assert E.gap_closed(0.5, 0.4, 0.3, higher_better=False)[0] == pytest.approx(0.5)
    assert E.gap_closed(0.5, 0.5, 0.3, False)[0] == pytest.approx(0.0)
    assert E.gap_closed(0.5, 0.3, 0.3, False)[0] == pytest.approx(1.0)
    assert E.gap_closed(0.5, 0.2, 0.3, False)[0] == pytest.approx(1.5)  # not clipped
    assert E.gap_closed(0.5, 0.6, 0.3, False)[0] == pytest.approx(-0.5)
    # higher is better: random 0.8, full 0.9, coreset 0.85 -> 0.5 (formula is sign-symmetric)
    assert E.gap_closed(0.8, 0.85, 0.9, higher_better=True)[0] == pytest.approx(0.5)


def test_gap_closed_degenerate_headroom():
    v, f = E.gap_closed(0.5, 0.4, 0.5 - 1e-4, False, noise=0.01)
    assert np.isnan(v) and f == "saturated"
    v, f = E.gap_closed(0.5, 0.4, 0.5, False)
    assert np.isnan(v) and f == "saturated"


def _sim_rows(n_datasets=8, n_seeds=4, effect=0.0, saturate=(), seed=0, tiers=None, effect_by_tier=None):
    """Simulated coreset results: full, random and `core` at budgets 100 and 400, plus a random learning curve."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_datasets):
        tier = tiers[d] if tiers else "small"
        eff = effect_by_tier.get(tier, 0.0) if effect_by_tier else effect
        ds_off = rng.normal(0, 0.05)
        full = 0.3 + ds_off
        for s in range(n_seeds):
            base = dict(dataset=f"d{d}", tier=tier, split="confirm", split_seed=0, seed=s, metric="log_loss", n_train=10_000, flags="", task="classification")
            full_v = full + rng.normal(0, 0.002)
            rows.append({**base, "method": "full", "budget": 10_000, "budget_key": "full", "ratio": 1.0, "value": full_v})
            for b in (50, 100, 200, 400, 800, 1600):
                # saturated datasets: random is indistinguishable from full (no headroom at any budget)
                r = full_v if f"d{d}" in saturate else full + 2.0 * b**-0.5 + rng.normal(0, 0.003)
                rows.append({**base, "method": "random", "budget": b, "budget_key": f"b{b}", "ratio": b / 10_000, "value": r})
                if b in (100, 400):
                    rows.append({**base, "method": "core", "budget": b, "budget_key": f"b{b}", "ratio": b / 10_000, "value": r - eff + rng.normal(0, 0.003)})
    return pd.DataFrame(rows)


def test_saturated_cells_excluded_from_claims():
    df = _sim_rows(effect=0.02, saturate={"d0", "d1"})
    flagged = E.annotate_saturation(df)
    assert flagged[(flagged["dataset"] == "d0") & (flagged["method"] == "core")]["flags"].str.contains("saturated").all()
    assert not flagged[(flagged["dataset"] == "d3") & (flagged["method"] == "core")]["flags"].str.contains("saturated").any()
    r = S.compare(flagged[flagged["budget"] == 100], "core", "random", "log_loss")
    assert set(r["excluded"]) == {"d0", "d1"} and r["n_datasets"] == 6


def test_paired_selection_beats_unpaired():
    """Simulated metric = dataset + split effect + split x method interaction + selection noise. Holding the
    split fixed and varying only the selection seed removes split-level variance from the paired differences."""
    rng = np.random.default_rng(0)

    def rows(vary_split, n=200):
        out = []
        split_eff = {k: rng.normal(0, 0.05) for k in range(n)}
        inter = {k: rng.normal(0, 0.02) for k in range(n)}
        for k in range(n):
            sp = k if vary_split else 0
            base = dict(dataset="d0", split_seed=sp, seed=k, metric="log_loss", flags="")
            r = 0.5 + split_eff[sp] + rng.normal(0, 0.01)
            c = 0.49 + split_eff[sp] + inter[sp] + rng.normal(0, 0.01)
            out += [{**base, "method": "random", "value": r}, {**base, "method": "core", "value": c}]
        return pd.DataFrame(out)

    def paired_var(df):
        m = df[df.method == "core"].set_index(["split_seed", "seed"])["value"]
        b = df[df.method == "random"].set_index(["split_seed", "seed"])["value"]
        return float((m - b).var())

    assert paired_var(rows(vary_split=False)) < paired_var(rows(vary_split=True))


def test_bootstrap_resamples_datasets():
    rng = np.random.default_rng(0)

    def width(D, S_):
        ds_eff = rng.normal(0.01, 0.02, D)
        rows = []
        for d in range(D):
            for s in range(S_):
                v = ds_eff[d] + rng.normal(0, 0.02)
                rows += [
                    {"dataset": f"d{d}", "seed": s, "split_seed": 0, "method": "b", "metric": "accuracy", "value": 0.5, "flags": ""},
                    {"dataset": f"d{d}", "seed": s, "split_seed": 0, "method": "m", "metric": "accuracy", "value": 0.5 + v, "flags": ""},
                ]
        r = S.compare(pd.DataFrame(rows), "m", "b", "accuracy", n_boot=1000)
        return r["ci_high"] - r["ci_low"]

    w = {(D, S_): np.mean([width(D, S_) for _ in range(15)]) for D in (10, 40) for S_ in (2, 20, 80)}
    # more seeds stop helping at the dataset-level floor
    assert w[(10, 80)] > 0.6 * w[(10, 20)]
    # 4x the datasets roughly halves the width
    assert w[(40, 20)] == pytest.approx(w[(10, 20)] / 2, rel=0.3)


@pytest.mark.slow
def test_type_I_error_controlled():
    rng = np.random.default_rng(1)
    n_exp, D = 1000, 12
    single = 0
    family = 0
    for i in range(n_exp):
        per_ds = rng.normal(0, 0.02, D)
        lo, hi = S.bootstrap_ci(per_ds, n_boot=200, seed=i)
        single += (lo > 0 or hi < 0) and S.sign_test(per_ds) < 0.05
        ps = [S.sign_test(rng.normal(0, 0.02, D)) for _ in range(6)]  # (method x budget) family of 6 null comparisons
        family += (S.holm(ps) < 0.05).any()
    assert single / n_exp <= 0.07
    assert family / n_exp <= 0.07


@pytest.mark.slow
def test_power_meets_design():
    sd_dataset, sd_seed, S_ = 0.01, 0.02, 10
    mde_target = 0.01
    D = 6
    while E.min_detectable_effect(D, S_, sd_dataset, sd_seed) > mde_target:
        D += 2
    assert E.simulate_power(D, S_, mde_target, sd_dataset, sd_seed, n_sim=400, seed=0) >= 0.8
    # an independent simulation agrees within Monte Carlo error
    assert E.simulate_power(D, S_, mde_target, sd_dataset, sd_seed, n_sim=400, seed=123) >= 0.75
    assert E.min_detectable_effect(5, S_, sd_dataset, sd_seed) == float("inf")  # sign test cannot reach p < 0.05


def test_min_detectable_effect_monotone():
    base = E.min_detectable_effect(10, 5, 0.01, 0.02)
    assert E.min_detectable_effect(20, 5, 0.01, 0.02) < base
    assert E.min_detectable_effect(10, 20, 0.01, 0.02) < base
    assert E.min_detectable_effect(10, 5, 0.02, 0.02) > base
    assert E.min_detectable_effect(10, 5, 0.01, 0.04) > base
    # noisier (small-tier) noise needs a larger effect
    assert E.min_detectable_effect(10, 5, 0.03, 0.06) > E.min_detectable_effect(10, 5, 0.01, 0.02)


def test_holm_correction():
    p = np.array([0.001, 0.02, 0.04, 0.3])
    adj = S.holm(p)
    np.testing.assert_allclose(adj, [0.004, 0.06, 0.08, 0.3])
    assert (adj >= p).all()
    assert (np.diff(adj[np.argsort(p)]) >= 0).all()
    assert S.holm([0.6, 0.7]).max() == 1.0


def test_heterogeneity_by_tier():
    tiers = ["small"] * 10 + ["large"] * 10
    df = _sim_rows(n_datasets=20, n_seeds=3, tiers=tiers, effect_by_tier={"small": 0.03, "large": 0.0}, seed=3)
    res = S.compare_by_tier(df[df["budget"] == 100], "core", "random", "log_loss")
    assert res["tiers"]["small"]["significant"]
    assert not res["tiers"]["large"]["significant"]
    assert res["heterogeneous"]


def test_ratio_curve_summary():
    df = _sim_rows(effect=0.01)
    eff = E.effect_sizes(df, "core", "log_loss")
    eff = pd.concat([eff, pd.DataFrame([{**eff.iloc[0].to_dict(), "dataset": "dx", "ratio": 0.5, "gap_closed": np.nan}])])
    curve = E.ratio_curve(eff)
    assert list(curve["ratio"]) == sorted(curve["ratio"]) and curve["ratio"].is_unique
    assert len(curve) == 3
    assert E.ratio_curve(pd.DataFrame()).empty
