"""Custom preprocessing applied *before* data reaches TabPFN.

TabPFN's cost grows with the size of the training context it sees:
attention runs across rows (train samples) and across features, so both
``n_train`` and ``n_features`` drive fit/predict time and GPU memory.
The steps here try to shrink that context, or the bytes copied to the
device, without hurting accuracy.

Two kinds of steps:

* Column transforms (sklearn ``TransformerMixin``) - safe to put in a
  ``Pipeline`` in front of ``TabPFNClassifier`` because they're applied
  identically to train and test.
* Row reducers (plain functions on ``X, y``) - they change the number of
  samples, which sklearn transformers are not allowed to do, so they're
  applied to the training set only, before ``fit``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class DropConstantColumns(TransformerMixin, BaseEstimator):
    """Drop columns whose variance on the training set is <= ``threshold``."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan")
        variances = np.nanvar(X, axis=0)
        self.keep_mask_ = ~(variances <= self.threshold)
        if not self.keep_mask_.any():
            # Never hand TabPFN an empty matrix.
            self.keep_mask_[0] = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return X[:, self.keep_mask_]


class DropCorrelatedColumns(TransformerMixin, BaseEstimator):
    """Greedily drop columns with |pearson r| above ``threshold`` to an earlier kept column."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan")
        filled = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(filled, rowvar=False))
        corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)

        keep: list[int] = []
        for j in range(X.shape[1]):
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.keep_mask_[keep] = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return X[:, self.keep_mask_]


class CastFloat32(TransformerMixin, BaseEstimator):
    """Cast to float32 so we never ship float64 tensors to the device."""

    def fit(self, X, y=None):
        validate_data(self, X, ensure_all_finite="allow-nan")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return X.astype(np.float32, copy=False)


def deduplicate_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove exact duplicate (row, label) pairs, keeping first occurrence order."""
    X = np.asarray(X)
    y = np.asarray(y)
    combined = np.column_stack([X, y]) if X.ndim == 2 else X
    _, first_idx = np.unique(combined, axis=0, return_index=True)
    first_idx.sort()
    return X[first_idx], y[first_idx]


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap the training set at ``max_rows`` while keeping class proportions.

    Every class keeps at least one row so TabPFN still sees all labels.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    if n <= max_rows:
        return X, y

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if max_rows < len(classes):
        raise ValueError(
            f"max_rows={max_rows} is smaller than the number of classes ({len(classes)})"
        )

    # Proportional allocation, at least one per class, then fix rounding.
    alloc = np.maximum(1, np.floor(counts / n * max_rows).astype(int))
    alloc = np.minimum(alloc, counts)
    while alloc.sum() > max_rows:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < max_rows:
        room = counts - alloc
        alloc[np.argmax(room)] += 1

    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == c), size=k, replace=False)
            for c, k in zip(classes, alloc)
        ]
    )
    idx.sort()
    return X[idx], y[idx]
