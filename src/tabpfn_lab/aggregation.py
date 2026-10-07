"""Aggregation rules over per-estimator outputs.

Classification aggregators: f(logits (E, n, C), temperature, **kw) -> (n, C) probabilities. All apply the
temperature to each estimator's logits first, exactly as the package does.

Regression aggregators: f(probs (E, n, B), dist, levels, **kw) -> RegPrediction (mean, quantiles, and
aggregated log-probabilities when the rule produces a distribution).

Adding a method = one function + one registry line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import trim_mean

from .backends import BarDistribution, log_softmax, softmax

DEFAULT_LEVELS = tuple(np.round(np.arange(0.025, 0.976, 0.025), 3))


def _renorm(p, fallback=None):
    """Renormalise rows; rows already summing to 1 (within 1e-12) are left bit-identical."""
    s = p.sum(axis=-1, keepdims=True)
    if fallback is not None:
        bad = s[..., 0] <= 1e-12
        if bad.any():
            p = p.copy()
            p[bad] = fallback[bad]
            s = p.sum(axis=-1, keepdims=True)
    return np.where(np.abs(s - 1.0) > 1e-12, p / s, p)


def per_estimator_proba(logits, temperature):
    return softmax(np.asarray(logits, np.float64) / temperature)


def _require_val(val_logits, val_y, name):
    if val_logits is None or val_y is None:
        raise ValueError(f"aggregator {name!r} needs val_logits and val_y from a train-only validation slice")


# --------------------------------------------------------------------------- classification


def agg_mean(logits, temperature, **_):
    return per_estimator_proba(logits, temperature).mean(axis=0)


def agg_logit_mean(logits, temperature, **_):
    return softmax(np.asarray(logits, np.float64).mean(axis=0) / temperature)


def agg_median(logits, temperature, **_):
    P = per_estimator_proba(logits, temperature)
    return _renorm(np.median(P, axis=0), fallback=P.mean(axis=0))


def agg_trimmed_mean(logits, temperature, trim=0.2, **_):
    P = per_estimator_proba(logits, temperature)
    return _renorm(trim_mean(P, trim, axis=0), fallback=P.mean(axis=0))


def agg_entropy_weighted(logits, temperature, tau=1.0, **_):
    P = per_estimator_proba(logits, temperature)
    H = -(P * np.log(np.clip(P, 1e-300, None))).sum(axis=-1)  # (E, n)
    w = softmax(-H / tau, axis=0)
    return _renorm((w[..., None] * P).sum(axis=0))


def agg_weighted(logits, temperature, w=None, **_):
    P = per_estimator_proba(logits, temperature)
    if w is None:
        raise ValueError("aggregator 'weighted' needs fixed estimator weights w")
    w = np.asarray(w, np.float64)
    if w.shape != (P.shape[0],) or (w < 0).any():
        raise ValueError(f"weights must be non-negative with shape ({P.shape[0]},), got {w.shape}")
    w = w / w.sum()
    return _renorm(np.tensordot(w, P, axes=1))


def estimator_log_losses(val_logits, val_y, temperature):
    P = per_estimator_proba(val_logits, temperature)
    y = np.asarray(val_y, int)
    p = np.clip(P[:, np.arange(len(y)), y], 1e-15, 1.0)
    return -np.log(p).mean(axis=1)


def fit_val_weights(val_logits, val_y, temperature, tau=0.1):
    """w_e proportional to exp(-logloss_e / tau), with log-losses measured on a train-only validation slice."""
    ll = estimator_log_losses(val_logits, val_y, temperature)
    return softmax(-(ll - ll.min()) / tau, axis=0)


def agg_val_weighted(logits, temperature, val_logits=None, val_y=None, tau=0.1, **_):
    _require_val(val_logits, val_y, "val_weighted")
    return agg_weighted(logits, temperature, w=fit_val_weights(val_logits, val_y, temperature, tau))


def _val_logloss(val_logits, val_y, temperature):
    P = agg_mean(val_logits, temperature)
    y = np.asarray(val_y, int)
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1.0)).mean())


def fit_temperature(val_logits, val_y, temperature, bounds=(0.05, 20.0)):
    """Multiplicative factor s for the temperature (T' = T * s) minimising validation log-loss of `mean`.

    s = 1 is always a candidate, so the fitted validation log-loss never exceeds the unfitted one.
    """
    f = lambda log_s: _val_logloss(val_logits, val_y, temperature * np.exp(log_s))  # noqa: E731
    grid = np.linspace(np.log(bounds[0]), np.log(bounds[1]), 25)
    best = min(grid, key=f)
    res = minimize_scalar(f, bounds=(best - 0.3, best + 0.3), method="bounded", options={"xatol": 1e-4})
    candidates = [0.0, best, float(res.x)]
    return float(np.exp(min(candidates, key=f)))


def agg_mean_temp_offline(logits, temperature, val_logits=None, val_y=None, **_):
    _require_val(val_logits, val_y, "mean_temp_offline")
    return agg_mean(logits, temperature * fit_temperature(val_logits, val_y, temperature))


CLF_AGGREGATORS = {
    "mean": agg_mean,
    "logit_mean": agg_logit_mean,
    "median": agg_median,
    "trimmed_mean": agg_trimmed_mean,
    "entropy_weighted": agg_entropy_weighted,
    "weighted": agg_weighted,
    "val_weighted": agg_val_weighted,
    "mean_temp_offline": agg_mean_temp_offline,
}
NEEDS_VALIDATION = {"val_weighted", "mean_temp_offline"}
NEEDS_WEIGHTS = {"weighted"}
# rules that intentionally change a single estimator's probabilities (temperature refit)
CHANGES_SINGLE_ESTIMATOR = {"mean_temp_offline"}


def get_clf_aggregator(name):
    try:
        return CLF_AGGREGATORS[name]
    except KeyError:
        raise KeyError(f"unknown classification aggregator {name!r}; available: {sorted(CLF_AGGREGATORS)}") from None


def aggregate_clf(name, logits, temperature, **kw) -> np.ndarray:
    return get_clf_aggregator(name)(logits, temperature, **kw)


# --------------------------------------------------------------------------- regression


@dataclass
class RegPrediction:
    mean: np.ndarray  # (n,)
    quantiles: np.ndarray  # (n, L)
    levels: np.ndarray  # (L,)
    logp: np.ndarray | None = None  # (n, B) aggregated log-probabilities when the rule yields a distribution


def point_from_log_probs(dist: BarDistribution, logp) -> np.ndarray:
    return dist.mean(logp)


def _from_logp(logp, dist, levels):
    levels = np.asarray(levels, float)
    return RegPrediction(point_from_log_probs(dist, logp), dist.quantiles(logp, levels), levels, logp)


def agg_mixture(probs, dist, levels=DEFAULT_LEVELS, **_):
    with np.errstate(divide="ignore"):
        logp = np.log(np.asarray(probs, np.float64).mean(axis=0))
    return _from_logp(logp, dist, levels)


def agg_log_pool(probs, dist, levels=DEFAULT_LEVELS, **_):
    with np.errstate(divide="ignore"):
        logp = log_softmax(np.log(np.asarray(probs, np.float64)).mean(axis=0))
    return _from_logp(logp, dist, levels)


def _per_estimator_quantiles(probs, dist, levels):
    with np.errstate(divide="ignore"):
        logp = np.log(np.asarray(probs, np.float64))
    return dist.quantiles(logp, levels), dist.mean(logp)  # (E, n, L), (E, n)


def agg_quantile_mean(probs, dist, levels=DEFAULT_LEVELS, **_):
    levels = np.asarray(levels, float)
    q, m = _per_estimator_quantiles(probs, dist, levels)
    return RegPrediction(m.mean(axis=0), q.mean(axis=0), levels)


def agg_quantile_median(probs, dist, levels=DEFAULT_LEVELS, **_):
    levels = np.asarray(levels, float)
    q, m = _per_estimator_quantiles(probs, dist, levels)
    return RegPrediction(np.median(m, axis=0), np.median(q, axis=0), levels)


REG_AGGREGATORS = {
    "mixture": agg_mixture,
    "log_pool": agg_log_pool,
    "quantile_mean": agg_quantile_mean,
    "quantile_median": agg_quantile_median,
}


def get_reg_aggregator(name):
    try:
        return REG_AGGREGATORS[name]
    except KeyError:
        raise KeyError(f"unknown regression aggregator {name!r}; available: {sorted(REG_AGGREGATORS)}") from None


def aggregate_reg(name, probs, dist, levels=DEFAULT_LEVELS, **kw) -> RegPrediction:
    return get_reg_aggregator(name)(probs, dist, levels=levels, **kw)
