import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

from tabpfn_lab import metrics as M


def _random_probs(n, C, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((n, C)) + 0.05
    return p / p.sum(1, keepdims=True), rng.integers(0, C, n)


def test_log_loss_matches_sklearn():
    p, y = _random_probs(200, 4)
    assert M.log_loss(y, p) == pytest.approx(sk_log_loss(y, p, labels=range(4)), abs=1e-9)


def test_log_loss_perfect_and_uniform():
    y = np.array([0, 1, 2, 1])
    assert M.log_loss(y, np.eye(3)[y]) == pytest.approx(0.0, abs=1e-12)
    assert M.log_loss(y, np.full((4, 3), 1 / 3)) == pytest.approx(np.log(3))


def test_accuracy_known_cases():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    assert M.accuracy([0, 1, 1, 1], p) == 0.75
    assert M.accuracy([0, 1, 0, 1], p) == 1.0


def test_roc_auc_matches_sklearn_binary_and_multiclass():
    p, y = _random_probs(300, 2, seed=1)
    assert M.roc_auc(y, p) == pytest.approx(roc_auc_score(y, p[:, 1]))
    p3, y3 = _random_probs(300, 3, seed=2)
    assert M.roc_auc(y3, p3) == pytest.approx(roc_auc_score(y3, p3, multi_class="ovr", average="macro"))


def test_roc_auc_constant_scores():
    y = np.array([0, 1, 0, 1, 1])
    assert M.roc_auc(y, np.full((5, 2), 0.5)) == pytest.approx(0.5)


def test_ece_bounds_and_calibration():
    rng = np.random.default_rng(0)
    n = 200_000
    conf = rng.uniform(0.5, 1.0, n)
    y = (rng.random(n) < conf).astype(int)  # class 1 with probability conf: perfectly calibrated
    p = np.column_stack([1 - conf, conf])
    e = M.ece(y, p)
    assert 0 <= e <= 0.02
    wrong = np.column_stack([np.full(n, 0.01), np.full(n, 0.99)])
    assert M.ece(np.zeros(n, int), wrong) > 0.9


def test_rmse_known_value():
    assert M.rmse([0, 0, 0, 0], [1, -1, 1, -1]) == pytest.approx(1.0)
    assert M.rmse([1, 2], [1, 4]) == pytest.approx(np.sqrt(2))


def test_crps_from_quantiles():
    levels = np.array([0.1, 0.5, 0.9])
    y = np.array([1.0, 2.0])
    exact = np.repeat(y[:, None], 3, axis=1)
    assert M.crps_from_quantiles(y, exact, levels) == 0.0
    near = exact + np.array([-0.5, 0.0, 0.5])
    far = near + 3.0
    assert M.crps_from_quantiles(y, far, levels) > M.crps_from_quantiles(y, near, levels)
    # hand computation for one row: y=0, quantiles (-1, 0, 2) at levels (0.1, 0.5, 0.9)
    # pinball: 0.1*(0-(-1)) = 0.1; 0; (0.9-1)*(0-2) = 0.2  -> mean 0.1 -> crps = 0.2
    assert M.crps_from_quantiles([0.0], [[-1.0, 0.0, 2.0]], levels) == pytest.approx(0.2)


def test_coverage_gaussian():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 50_000)
    lo, hi = norm.ppf(0.1), norm.ppf(0.9)
    assert M.coverage(y, np.full_like(y, lo), np.full_like(y, hi)) == pytest.approx(0.8, abs=0.03)


def test_metric_direction_registry():
    for m in M.CLF_METRICS + M.REG_METRICS + ["brier", "nll", "coverage"]:
        assert M.METRIC_DIRECTION[m] in ("lower", "higher")
    assert M.higher_is_better("accuracy") and not M.higher_is_better("log_loss")
    with pytest.raises(KeyError):
        M.higher_is_better("nope")
