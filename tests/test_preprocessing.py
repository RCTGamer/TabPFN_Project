"""Fast unit tests for the custom preprocessing - no TabPFN weights needed."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from tabpfn_lab.preprocessing import (
    CastFloat32,
    DropConstantColumns,
    DropCorrelatedColumns,
    deduplicate_rows,
    stratified_subsample,
)


def test_drop_constant_columns_removes_only_constants():
    X = np.array([[1.0, 5.0, 0.0], [2.0, 5.0, 1.0], [3.0, 5.0, 0.0]])
    out = DropConstantColumns().fit_transform(X)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out, X[:, [0, 2]])


def test_drop_constant_columns_uses_train_mask_on_test():
    X_train = np.array([[1.0, 5.0], [2.0, 5.0]])
    X_test = np.array([[9.0, 7.0]])  # column 1 varies here, but was constant in train
    step = DropConstantColumns().fit(X_train)
    assert step.transform(X_test).shape == (1, 1)


def test_drop_constant_columns_never_returns_empty():
    X = np.ones((4, 3))
    assert DropConstantColumns().fit_transform(X).shape == (4, 1)


def test_drop_correlated_columns_keeps_first_of_each_group():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 200))
    X = np.column_stack([a, 2 * a + 1, b, -b])
    step = DropCorrelatedColumns(threshold=0.95).fit(X)
    np.testing.assert_array_equal(step.keep_mask_, [True, False, True, False])


def test_cast_float32():
    out = CastFloat32().fit_transform(np.arange(6, dtype=np.float64).reshape(3, 2))
    assert out.dtype == np.float32


def test_deduplicate_rows_keeps_distinct_labels():
    X = np.array([[1, 2], [1, 2], [1, 2], [3, 4]])
    y = np.array([0, 0, 1, 1])
    X_out, y_out = deduplicate_rows(X, y)
    # (row,label) pair [1,2]/0 is duplicated; [1,2]/1 is a distinct sample.
    assert len(y_out) == 3
    np.testing.assert_array_equal(y_out, [0, 1, 1])


@pytest.mark.parametrize("max_rows", [10, 50, 99])
def test_stratified_subsample_caps_rows_and_keeps_all_classes(max_rows):
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    X = np.arange(len(y))[:, None].astype(float)
    X_out, y_out = stratified_subsample(X, y, max_rows=max_rows)
    assert len(y_out) == max_rows
    assert set(y_out) == {0, 1, 2}
    # Rows stay aligned with their labels.
    np.testing.assert_array_equal(y[X_out[:, 0].astype(int)], y_out)


def test_stratified_subsample_is_noop_when_small_enough():
    X, y = np.zeros((5, 2)), np.array([0, 1, 0, 1, 0])
    X_out, y_out = stratified_subsample(X, y, max_rows=10)
    assert X_out is X and y_out is y


def test_column_steps_compose_in_pipeline(redundant_split):
    X_train, X_test, y_train, _ = redundant_split
    pipe = make_pipeline(DropConstantColumns(), DropCorrelatedColumns(0.99), CastFloat32())
    out_train = clone(pipe).fit_transform(X_train, y_train)
    pipe.fit(X_train, y_train)
    out_test = pipe.transform(X_test)
    # 30 original cols + 5 constant + 10 near-copies -> constants and copies removed.
    assert out_train.shape[1] == out_test.shape[1] <= 30
    assert out_test.dtype == np.float32
