"""Baseline TabPFN vs. TabPFN behind custom preprocessing.

The contract: preprocessing may make things faster / leaner, but must not
cost more than ``ACCURACY_TOLERANCE`` accuracy. Timings and GPU memory are
printed (run with ``-s``) rather than asserted, since they're noisy across
machines - except on CUDA, where a smaller training context must not use
*more* peak memory than the baseline.
"""

import pytest
from sklearn.pipeline import make_pipeline

from tabpfn_lab.benchmark import run_benchmark
from tabpfn_lab.preprocessing import (
    CastFloat32,
    DropConstantColumns,
    DropCorrelatedColumns,
    deduplicate_rows,
    stratified_subsample,
)

pytestmark = pytest.mark.model

ACCURACY_TOLERANCE = 0.03
N_ESTIMATORS = 2  # keep CPU runs short; raise for real experiments


def make_tabpfn(device):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(device=device, n_estimators=N_ESTIMATORS, random_state=0)


def _report(*results):
    for r in results:
        print(
            f"\n{r.name:>12}: rows={r.n_train_rows:4d} feats={r.n_features_in:3d} "
            f"fit={r.fit_seconds:6.2f}s predict={r.predict_seconds:6.2f}s "
            f"acc={r.accuracy:.3f} gpu_fit={r.peak_gpu_mb_fit} gpu_pred={r.peak_gpu_mb_predict}"
        )


def test_tabpfn_baseline_is_accurate(tabpfn_available, device, breast_cancer_split):
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = run_benchmark("baseline", make_tabpfn(device), X_train, y_train, X_test, y_test)
    _report(result)
    assert result.accuracy > 0.9


def test_preprocessing_shrinks_context_without_hurting_accuracy(
    tabpfn_available, device, redundant_split
):
    X_train, X_test, y_train, y_test = redundant_split

    baseline = run_benchmark("baseline", make_tabpfn(device), X_train, y_train, X_test, y_test)

    preprocessed_model = make_pipeline(
        DropConstantColumns(),
        DropCorrelatedColumns(threshold=0.99),
        CastFloat32(),
        make_tabpfn(device),
    )
    preprocessed = run_benchmark(
        "preprocessed",
        preprocessed_model,
        X_train,
        y_train,
        X_test,
        y_test,
        row_reducers=[deduplicate_rows],
    )
    _report(baseline, preprocessed)

    # The custom steps actually shrank what TabPFN sees.
    tabpfn = preprocessed_model[-1]
    assert preprocessed.n_train_rows < baseline.n_train_rows
    assert tabpfn.n_features_in_ < X_train.shape[1]

    assert preprocessed.accuracy >= baseline.accuracy - ACCURACY_TOLERANCE

    if device == "cuda":
        assert preprocessed.peak_gpu_mb_fit <= baseline.peak_gpu_mb_fit * 1.05


@pytest.mark.parametrize("max_rows", [100, 200])
def test_subsampled_context_stays_close_to_baseline(
    tabpfn_available, device, breast_cancer_split, max_rows
):
    X_train, X_test, y_train, y_test = breast_cancer_split

    baseline = run_benchmark("baseline", make_tabpfn(device), X_train, y_train, X_test, y_test)
    subsampled = run_benchmark(
        f"rows<={max_rows}",
        make_tabpfn(device),
        X_train,
        y_train,
        X_test,
        y_test,
        row_reducers=[lambda X, y: stratified_subsample(X, y, max_rows=max_rows)],
    )
    _report(baseline, subsampled)

    assert subsampled.n_train_rows == max_rows
    # Smaller context trades some accuracy; allow a wider margin here.
    assert subsampled.accuracy >= baseline.accuracy - 2 * ACCURACY_TOLERANCE
