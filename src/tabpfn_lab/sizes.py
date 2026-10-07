"""Size tiers, budget resolution, and synthetic generators (data and learning curves with known parameters)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("tabpfn_lab")

# n_train ranges, lower bound inclusive
TIERS = {
    "tiny": (200, 1_000),
    "small": (1_000, 10_000),
    "medium": (10_000, 100_000),
    "large": (100_000, 1_000_000),
}
TIER_ORDER = ["tiny", "small", "medium", "large"]
REDUCTION_RATIOS = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
DEFAULT_ABSOLUTE_BUDGETS = [500, 2000, 8000, 32000]


def tier_of(n_train: int) -> str:
    """Tier for a training-set size. Sizes below the tiny range count as tiny, above the large range as large."""
    for name in reversed(TIER_ORDER):
        if n_train >= TIERS[name][0]:
            return name
    return "tiny"


def resolve_budgets(n_train, fractions=None, absolute=None, n_classes=None) -> list[int]:
    """Union of round(f * n_train) and absolute budgets.

    Budgets are clipped up to max(n_classes, 2) (logged), budgets >= n_train are dropped
    (that cell is the `full` reference), and the result is unique and sorted.
    """
    floor = max(int(n_classes or 0), 2)
    raw = [int(round(f * n_train)) for f in (fractions or [])] + [int(b) for b in (absolute or [])]
    out = set()
    for b in raw:
        if b < floor:
            log.info("budget %d below minimum %d for n_train=%d; clipped up", b, floor, n_train)
            b = floor
        if b >= n_train:
            log.info("budget %d >= n_train=%d; dropped (covered by the full reference)", b, n_train)
            continue
        out.add(b)
    return sorted(out)


def budget_keys(n_train, fractions=None, absolute=None, n_classes=None) -> dict[str, int]:
    """Like `resolve_budgets` but keeps the label of each budget ("f0.1", "a500"). Duplicate budgets keep the first label."""
    keys: dict[str, int] = {}
    seen: set[int] = set()
    for f in fractions or []:
        b = resolve_budgets(n_train, [f], None, n_classes)
        if b and b[0] not in seen:
            keys[f"f{f:g}"] = b[0]
            seen.add(b[0])
    for a in absolute or []:
        b = resolve_budgets(n_train, None, [a], n_classes)
        if b and b[0] not in seen:
            keys[f"a{int(a)}"] = b[0]
            seen.add(b[0])
    return keys


def learning_curve_budgets(n_train, n_points=6, n_classes=None, low=None, high_frac=0.5) -> list[int]:
    """At least `n_points` log-spaced budgets between a small floor and high_frac * n_train (for the random learning curve)."""
    lo = max(int(low or 0), max(int(n_classes or 0), 2) * 5, 10)
    hi = max(int(high_frac * n_train), lo + n_points)
    hi = min(hi, n_train - 1)
    grid = np.unique(np.round(np.geomspace(lo, hi, n_points)).astype(int))
    k = n_points
    while len(grid) < n_points and k < 10 * n_points:
        k += 1
        grid = np.unique(np.round(np.geomspace(lo, hi, k)).astype(int))
    return [int(b) for b in grid if b < n_train]


def make_synthetic(
    n,
    n_features,
    n_classes=2,
    task="classification",
    imbalance=None,
    noise=0.1,
    seed=0,
    heavy_tail=False,
):
    """Fast synthetic tabular data.

    Classification: each class is a Gaussian blob around a random centre, plus label noise `noise`.
    `imbalance` is the fraction of rows in the last (minority) class, or a list of class proportions.
    Regression: a smooth nonlinear function of the first features plus Gaussian noise; `heavy_tail`
    switches to Student-t (df=2) noise.
    """
    rng = np.random.default_rng(seed)
    if task == "classification":
        if imbalance is None:
            probs = np.full(n_classes, 1.0 / n_classes)
        elif np.ndim(imbalance) == 1:
            probs = np.asarray(imbalance, float) / np.sum(imbalance)
            n_classes = len(probs)
        else:
            rest = (1.0 - imbalance) / max(n_classes - 1, 1)
            probs = np.array([rest] * (n_classes - 1) + [imbalance])
        counts = rng.multinomial(n, probs)
        # every class gets at least 2 rows so stratified splits work
        for c in range(n_classes):
            while counts[c] < 2:
                j = int(np.argmax(counts))
                counts[j] -= 1
                counts[c] += 1
        y = np.repeat(np.arange(n_classes), counts)
        centres = rng.normal(0, 1.5, size=(n_classes, n_features))
        X = centres[y] + rng.normal(0, 1.0, size=(n, n_features))
        flip = rng.random(n) < noise
        y = y.copy()
        y[flip] = rng.integers(0, n_classes, flip.sum())
        perm = rng.permutation(n)
        return X[perm], y[perm]
    if task == "regression":
        X = rng.normal(0, 1, size=(n, n_features))
        w = rng.normal(0, 1, size=n_features)
        f = X @ w / np.sqrt(n_features) + np.sin(2 * X[:, 0]) + 0.5 * X[:, min(1, n_features - 1)] ** 2
        eps = rng.standard_t(2, size=n) if heavy_tail else rng.normal(0, 1, size=n)
        return X, f + noise * eps * (5.0 if heavy_tail else 1.0)
    raise ValueError(f"unknown task {task!r}")


@dataclass(frozen=True)
class LearningCurve:
    """Known power-law learning curve m(b) = m_inf + a * b**-alpha (lower is better when a > 0)."""

    m_inf: float
    a: float
    alpha: float

    def metric(self, budget) -> np.ndarray:
        return self.m_inf + self.a * np.asarray(budget, float) ** (-self.alpha)

    def sample(self, budgets, noise=0.0, seed=0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m = self.metric(budgets)
        return m + rng.normal(0, noise, size=np.shape(m))
