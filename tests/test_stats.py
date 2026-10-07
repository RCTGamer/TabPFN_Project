import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from tabpfn_lab import stats as S


def _df(diffs_by_dataset, metric="log_loss", base=1.0):
    """Long-format rows: for each dataset, one (method, baseline) pair per seed."""
    rows = []
    for d, diffs in diffs_by_dataset.items():
        for s, dv in enumerate(diffs):
            rows.append({"dataset": d, "seed": s, "split_seed": 0, "method": "b", "metric": metric, "value": base, "flags": ""})
            rows.append({"dataset": d, "seed": s, "split_seed": 0, "method": "m", "metric": metric, "value": base + dv, "flags": ""})
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_bootstrap_ci_covers_true_effect():
    rng = np.random.default_rng(0)
    hits = 0
    for i in range(200):
        x = rng.normal(0.3, 1.0, 30)
        lo, hi = S.bootstrap_ci(x, n_boot=1000, seed=i)
        hits += lo <= 0.3 <= hi
    assert hits / 200 >= 0.9


@pytest.mark.slow
def test_null_effect_rarely_significant():
    rng = np.random.default_rng(1)
    sig = 0
    for i in range(200):
        df = _df({f"d{j}": rng.normal(0, 0.1, 3) for j in range(15)})
        sig += S.compare(df, "m", "b", "log_loss", n_boot=500, seed=i)["significant"]
    assert sig / 200 <= 0.10


def test_sign_test_known_p():
    d = np.array([1] * 9 + [-1])
    expected = 2 * binom.sf(8, 10, 0.5)  # P(X >= 9) two-sided
    assert S.sign_test(d) == pytest.approx(expected)
    assert S.sign_test(d) == pytest.approx(22 / 1024)


def test_all_zero_differences_handled():
    assert S.wilcoxon_p(np.zeros(10)) == 1.0
    assert S.sign_test(np.zeros(10)) == 1.0


def test_win_rate_ties_half():
    assert S.win_rate([1, -1, 0, 0]) == pytest.approx(0.5)
    assert S.win_rate([1, 1, 0, -1]) == pytest.approx(0.625)


def test_dataset_weighting():
    # dataset A: 10 seeds of diff -1, dataset B: 1 seed of diff +3. Dataset-level mean = (-1 + 3) / 2 = 1
    df = _df({"A": [-1.0] * 10, "B": [3.0]})
    r = S.compare(df, "m", "b", "log_loss")
    assert r["mean_diff"] == pytest.approx(1.0)
    manual = pd.Series({"A": -1.0, "B": 3.0}).mean()
    assert r["mean_diff"] == pytest.approx(manual)


def test_direction_aware_compare():
    df = _df({f"d{i}": [-0.1, -0.1] for i in range(8)}, metric="log_loss")
    r = S.compare(df, "m", "b", "log_loss")
    assert r["mean_diff"] < 0 and r["improvement"] > 0 and r["win_rate"] == 1.0 and r["significant"]
    df_acc = _df({f"d{i}": [-0.1, -0.1] for i in range(8)}, metric="accuracy", base=0.8)
    r2 = S.compare(df_acc, "m", "b", "accuracy")
    assert r2["improvement"] < 0 and r2["win_rate"] == 0.0


def test_holm_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    adj = S.holm(p)
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert (adj >= np.array(p)).all() and adj.max() <= 1
    np.testing.assert_allclose(S.holm([0.5, 0.9]), [1.0, 1.0])
