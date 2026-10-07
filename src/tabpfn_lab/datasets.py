"""Dataset loaders (synthetic + OpenML) and the train/test split used by every experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.model_selection import train_test_split

from .sizes import make_synthetic


@dataclass
class Split:
    name: str
    task: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_seed: int
    classes: np.ndarray | None = None  # integer class codes 0..C-1 (classification)
    label_names: np.ndarray | None = None  # original labels, indexed by code
    meta: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def n_classes(self) -> int | None:
        return None if self.classes is None else len(self.classes)

    @property
    def test_hash(self) -> str:
        return hashlib.sha1(np.asarray(self.test_idx, dtype=np.int64).tobytes()).hexdigest()[:16]


def encode_labels(y):
    """Map arbitrary labels to codes 0..C-1. Returns (codes, label_names)."""
    names, codes = np.unique(np.asarray(y), return_inverse=True)
    return codes.astype(np.int64), names


def make_split(X, y, task, test_size=0.3, seed=0, name="") -> Split:
    """Seeded train/test split; stratified for classification when every class has >= 2 rows."""
    X = np.asarray(X)
    n = len(X)
    classes = label_names = None
    if task == "classification":
        y, label_names = encode_labels(y)
        classes = np.arange(len(label_names))
        stratify = y if np.bincount(y).min() >= 2 else None
    else:
        y = np.asarray(y, dtype=float)
        stratify = None
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)
    return Split(
        name=name,
        task=task,
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        train_idx=train_idx,
        test_idx=test_idx,
        split_seed=seed,
        classes=classes,
        label_names=label_names,
    )


def dataframe_to_numeric(df):
    """Numeric matrix from a DataFrame: numeric columns as float, everything else as category codes (NaN kept)."""
    import pandas as pd

    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
        else:
            codes = s.astype("category").cat.codes.to_numpy().astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
    return np.column_stack(cols) if cols else np.empty((len(df), 0))


def _parse_imbalance(v):
    if v in (None, "None"):
        return None
    if isinstance(v, str) and v.startswith("["):
        return [float(x) for x in v.strip("[]").split(",")]
    return float(v)


def _spec_key(spec: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in spec.items() if k not in ("tier", "split")))


@lru_cache(maxsize=64)
def _load_cached(key: tuple):
    spec = dict(key)
    source = spec.get("source", "synthetic")
    task = spec.get("task", "classification")
    if source == "synthetic":
        X, y = make_synthetic(
            n=int(spec.get("n", 500)),
            n_features=int(spec.get("n_features", 6)),
            n_classes=int(spec.get("n_classes", 2)),
            task=task,
            imbalance=_parse_imbalance(spec.get("imbalance")),
            noise=float(spec.get("noise", 0.1)),
            seed=int(spec.get("data_seed", 0)),
            heavy_tail=str(spec.get("heavy_tail", False)) == "True",
        )
    elif source == "openml":
        from sklearn.datasets import fetch_openml

        kwargs = {"as_frame": True, "parser": "auto"}
        if "data_id" in spec:
            bunch = fetch_openml(data_id=int(spec["data_id"]), **kwargs)
        else:
            bunch = fetch_openml(name=spec["name"], version=spec.get("version", "active"), **kwargs)
        X = dataframe_to_numeric(bunch.data)
        y = bunch.target.to_numpy()
        if task == "regression":
            y = y.astype(float)
        else:
            y = y.astype(str)
        keep = ~(np.isnan(y) if task == "regression" else (y == "nan"))
        X, y = X[keep], y[keep]
    else:
        raise ValueError(f"unknown dataset source {source!r}")
    max_rows = spec.get("max_rows")
    if max_rows not in (None, "None") and len(X) > int(max_rows):
        rng = np.random.default_rng(int(spec.get("data_seed", 0)))
        sel = np.sort(rng.choice(len(X), int(max_rows), replace=False))
        X, y = X[sel], y[sel]
    X.setflags(write=False)
    return X, y, task


def load_dataset(spec: dict):
    """Load (X, y, task) from a dataset spec dict, e.g. {name, source: synthetic|openml, task, ...}."""
    return _load_cached(_spec_key(spec))


def load_split(spec: dict, split_seed: int, test_size: float = 0.3) -> Split:
    X, y, task = load_dataset(spec)
    s = make_split(np.array(X), y, task, test_size=test_size, seed=split_seed, name=spec["name"])
    return s
