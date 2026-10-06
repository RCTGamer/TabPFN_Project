"""Measure fit/predict time, peak GPU memory and accuracy for one configuration."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score

RowReducer = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class BenchmarkResult:
    name: str
    n_train_rows: int
    n_features_in: int
    fit_seconds: float
    predict_seconds: float
    accuracy: float
    # None when not running on CUDA.
    peak_gpu_mb_fit: float | None = None
    peak_gpu_mb_predict: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _sync_and_reset_peak(use_cuda: bool) -> None:
    if use_cuda:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _sync_and_read_peak_mb(use_cuda: bool) -> float | None:
    if not use_cuda:
        return None
    import torch

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def run_benchmark(
    name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    row_reducers: list[RowReducer] | None = None,
    track_gpu: bool | None = None,
) -> BenchmarkResult:
    """Fit ``estimator`` (any sklearn-style model / Pipeline) and time it.

    ``row_reducers`` run on the training set only, before fit, and their
    cost is included in ``fit_seconds`` since it's part of the pipeline.
    """
    use_cuda = _cuda_available() if track_gpu is None else track_gpu

    _sync_and_reset_peak(use_cuda)
    t0 = time.perf_counter()
    for reduce in row_reducers or []:
        X_train, y_train = reduce(X_train, y_train)
    estimator.fit(X_train, y_train)
    peak_fit = _sync_and_read_peak_mb(use_cuda)
    fit_seconds = time.perf_counter() - t0

    _sync_and_reset_peak(use_cuda)
    t0 = time.perf_counter()
    y_pred = estimator.predict(X_test)
    peak_predict = _sync_and_read_peak_mb(use_cuda)
    predict_seconds = time.perf_counter() - t0

    return BenchmarkResult(
        name=name,
        n_train_rows=len(y_train),
        n_features_in=np.asarray(X_train).shape[1],
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        accuracy=float(accuracy_score(y_test, y_pred)),
        peak_gpu_mb_fit=peak_fit,
        peak_gpu_mb_predict=peak_predict,
    )
