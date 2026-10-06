"""Tests for the measurement harness itself, using a cheap sklearn model."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabpfn_lab.benchmark import run_benchmark
from tabpfn_lab.preprocessing import deduplicate_rows


def test_run_benchmark_reports_sane_numbers(redundant_split):
    X_train, X_test, y_train, y_test = redundant_split
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    result = run_benchmark(
        "logreg+dedup",
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        row_reducers=[deduplicate_rows],
        track_gpu=False,
    )

    assert result.n_train_rows == len(y_train) - 100  # the 100 duplicated rows are gone
    assert result.n_features_in == X_train.shape[1]
    assert result.fit_seconds > 0 and result.predict_seconds > 0
    assert 0.9 < result.accuracy <= 1.0
    assert result.peak_gpu_mb_fit is None
