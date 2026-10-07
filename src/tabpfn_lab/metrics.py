"""Scoring functions. Classification metrics take integer labels 0..C-1 and an (n, C) probability matrix."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-15

# direction of every metric the harness can emit: "lower" or "higher" is better
METRIC_DIRECTION = {
    "log_loss": "lower",
    "accuracy": "higher",
    "roc_auc": "higher",
    "ece": "lower",
    "brier": "lower",
    "rmse": "lower",
    "nll": "lower",
    "crps": "lower",
    "coverage": "higher",
}

# valid value ranges (inclusive); None means unbounded
METRIC_RANGE = {
    "log_loss": (0.0, None),
    "accuracy": (0.0, 1.0),
    "roc_auc": (0.0, 1.0),
    "ece": (0.0, 1.0),
    "brier": (0.0, 2.0),
    "rmse": (0.0, None),
    "nll": (None, None),
    "crps": (0.0, None),
    "coverage": (0.0, 1.0),
}

CLF_METRICS = ["log_loss", "accuracy", "roc_auc", "ece"]
REG_METRICS = ["rmse", "crps"]


def higher_is_better(metric: str) -> bool:
    if metric not in METRIC_DIRECTION:
        raise KeyError(f"unknown metric {metric!r}; known: {sorted(METRIC_DIRECTION)}")
    return METRIC_DIRECTION[metric] == "higher"


def log_loss(y, proba) -> float:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float)[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def accuracy(y, proba) -> float:
    return float(np.mean(np.argmax(proba, axis=1) == np.asarray(y)))


def roc_auc(y, proba) -> float:
    """Binary AUC on the positive column, or unweighted one-vs-rest macro AUC (same as sklearn's 'ovr'/'macro')."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] == 2:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, proba[:, 1]))
    aucs = []
    for c in range(proba.shape[1]):
        pos = y == c
        if pos.all() or not pos.any():
            continue
        aucs.append(roc_auc_score(pos.astype(int), proba[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def ece(y, proba, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y)).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(conf))


def brier(y, proba) -> float:
    proba = np.asarray(proba, dtype=float)
    onehot = np.eye(proba.shape[1])[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2)))


def nll(log_density) -> float:
    """Mean negative log density, given the predictive log density evaluated at each true target."""
    return float(-np.mean(log_density))


def pinball(y, q, level: float) -> np.ndarray:
    diff = np.asarray(y, float) - np.asarray(q, float)
    return np.maximum(level * diff, (level - 1) * diff)


def crps_from_quantiles(y, quantiles, levels) -> float:
    """CRPS approximated as 2 * the mean pinball loss over the given quantile levels.

    `quantiles` has shape (n, L) and `levels` length L.
    """
    quantiles = np.asarray(quantiles, float)
    levels = np.asarray(levels, float)
    y = np.asarray(y, float)[:, None]
    diff = y - quantiles
    loss = np.maximum(levels * diff, (levels - 1) * diff)
    return float(2.0 * loss.mean())


def coverage(y, lower, upper) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def score_classification(y, proba) -> dict[str, float]:
    return {
        "log_loss": log_loss(y, proba),
        "accuracy": accuracy(y, proba),
        "roc_auc": roc_auc(y, proba),
        "ece": ece(y, proba),
    }


def score_regression(y, mean, quantiles, levels) -> dict[str, float]:
    return {"rmse": rmse(y, mean), "crps": crps_from_quantiles(y, quantiles, levels)}
