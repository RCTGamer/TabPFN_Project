"""Benchmark-ready scikit-learn estimators: a thin shell over Backend + aggregation + coresets.

No logic is duplicated: selection, aggregation and validation-slice fitting all call the same functions the
experiments use. Anything an aggregator or coreset needs is carved out of the data passed to `fit`.
Benchmark adapters live in `benchmarks/`, never in this package.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from . import aggregation as agg
from . import coresets as cs
from .backends import Backend, SklearnBackend, TabPFNBackend


class _TableEncoder:
    """Numeric view of a table for selection / numeric-only backends; categories are learned in `fit`."""

    def fit(self, X):
        import pandas as pd

        self.is_frame_ = isinstance(X, pd.DataFrame)
        df = X if self.is_frame_ else pd.DataFrame(np.asarray(X, dtype=object) if np.asarray(X).dtype == object else np.asarray(X))
        self.columns_ = list(df.columns)
        self.categories_ = {}
        for c in df.columns:
            s = df[c]
            if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
                vals = pd.Series(s.dropna().unique())
                self.categories_[c] = list(vals.astype(str).sort_values())
        return self

    def transform(self, X) -> np.ndarray:
        import pandas as pd

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=object) if np.asarray(X).dtype == object else np.asarray(X))
        if list(df.columns) != self.columns_:
            df.columns = self.columns_ if len(df.columns) == len(self.columns_) else df.columns
        cols = []
        for c in self.columns_:
            s = df[c]
            if c in self.categories_:
                lookup = {v: i for i, v in enumerate(self.categories_[c])}
                cols.append(np.array([lookup.get(str(v), np.nan) if not pd.isna(v) else np.nan for v in s], dtype=float))
            else:
                cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.empty((len(df), 0))


def _take(X, idx):
    import pandas as pd

    return X.iloc[idx] if isinstance(X, pd.DataFrame) else np.asarray(X)[idx]


class _TabPFNLabBase(BaseEstimator):
    _task = "classification"
    _default_aggregator = "mean"

    def __init__(
        self,
        version="3.5",
        n_estimators=8,
        aggregator=None,
        aggregator_kwargs=None,
        coreset="none",
        budget=None,
        budget_fraction=None,
        coreset_kwargs=None,
        random_state=0,
        device="cuda",
        backend="tabpfn",
        val_fraction=0.2,
        tabpfn_kwargs=None,
    ):
        self.version = version
        self.n_estimators = n_estimators
        self.aggregator = aggregator
        self.aggregator_kwargs = aggregator_kwargs
        self.coreset = coreset
        self.budget = budget
        self.budget_fraction = budget_fraction
        self.coreset_kwargs = coreset_kwargs
        self.random_state = random_state
        self.device = device
        self.backend = backend
        self.val_fraction = val_fraction
        self.tabpfn_kwargs = tabpfn_kwargs

    # ---------------------------------------------------------------- helpers

    def _make_backend(self) -> Backend:
        if isinstance(self.backend, Backend):
            return self.backend
        if self.backend == "tabpfn":
            return TabPFNBackend(version=self.version, device=self.device, **(self.tabpfn_kwargs or {}))
        if self.backend == "sklearn":
            return SklearnBackend()
        raise ValueError(f"unknown backend {self.backend!r}")

    def _backend_view(self, X, X_num):
        return X if self.backend_.accepts_raw else X_num

    def resolve_budget(self, n) -> int | None:
        """Budget in rows for a fit on n rows, or None for no reduction."""
        if self.coreset in (None, "none"):
            return None
        if self.budget is not None:
            b = int(self.budget)
        elif self.budget_fraction is not None:
            b = max(1, int(round(float(self.budget_fraction) * n)))
        else:
            raise ValueError("coreset given but neither budget nor budget_fraction is set")
        return None if b >= n else b

    def _select_context(self, X, X_num, y):
        """Context rows chosen from the fit data only."""
        n = len(y)
        b = self.resolve_budget(n)
        self.budget_ = n if b is None else b
        Xb = self._backend_view(X, X_num)
        if b is None:
            self.context_ = [(Xb, y)]
            self.perest_ = False
            return
        kw = dict(self.coreset_kwargs or {})
        if self.coreset in cs.PER_ESTIMATOR_METHODS:
            subsets = cs.per_estimator_subsets(self.coreset, X_num, y, b, self.n_estimators, self.random_state, self._task)
            self.context_ = [(_take(Xb, s), y[s]) for s in subsets]
            self.perest_ = True
            return
        if self.coreset == "embedding_kmeans":
            kw.setdefault("backend", self.backend_)
        idx = cs.select_coreset(self.coreset, X_num, y, b, self.random_state, self._task, **kw)
        self.context_ = [(_take(Xb, idx), y[idx])]
        self.perest_ = False

    def _outputs(self, X):
        from .experiments import _stack_outputs

        Xb = self._backend_view(X, self.encoder_.transform(X))
        rs = int(self.random_state)
        if self.perest_:
            outs = [self._fit_predict(Xc, yc, Xb, 1, rs * 1000 + e) for e, (Xc, yc) in enumerate(self.context_)]
            return _stack_outputs(outs, self._task)
        Xc, yc = self.context_[0]
        return self._fit_predict(Xc, yc, Xb, self.n_estimators, rs)

    def _fit_common(self, X, y):
        self.backend_ = self._make_backend()
        self.encoder_ = _TableEncoder().fit(X)
        X_num = self.encoder_.transform(X)
        self.n_features_in_ = X_num.shape[1]
        self.aggregator_ = self.aggregator or self._default_aggregator
        self.agg_params_ = dict(self.aggregator_kwargs or {})
        self.temperature_factor_ = 1.0
        return X_num

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class TabPFNLabClassifier(ClassifierMixin, _TabPFNLabBase):
    _task = "classification"
    _default_aggregator = "mean"

    def _fit_predict(self, Xc, yc, X, n_estimators, seed):
        return self.backend_.clf_outputs(Xc, yc, X, self.class_codes_, n_estimators, seed)

    def fit(self, X, y):
        X_num = self._fit_common(X, y)
        self.classes_, y_enc = np.unique(np.asarray(y), return_inverse=True)
        self.class_codes_ = np.arange(len(self.classes_))
        agg.get_clf_aggregator(self.aggregator_)
        if self.aggregator_ in agg.NEEDS_VALIDATION:
            self._fit_on_validation_slice(X, X_num, y_enc)
        self._select_context(X, X_num, y_enc)
        return self

    def _fit_on_validation_slice(self, X, X_num, y_enc):
        """Fit aggregator weights / temperature on a validation slice of the fit data (never predict inputs)."""
        n = len(y_enc)
        strat = y_enc if np.bincount(y_enc).min() >= 2 else None
        tr, va = train_test_split(np.arange(n), test_size=self.val_fraction, random_state=self.random_state, stratify=strat)
        Xb = self._backend_view(X, X_num)
        val = self.backend_.clf_outputs(_take(Xb, tr), y_enc[tr], _take(Xb, va), self.class_codes_, self.n_estimators, int(self.random_state))
        tau = self.agg_params_.get("tau", 0.1)
        if self.aggregator_ == "val_weighted":
            self.agg_params_ = {"w": agg.fit_val_weights(val.logits, y_enc[va], val.temperature, tau)}
            self.aggregator_ = "weighted"
        else:  # mean_temp_offline
            self.temperature_factor_ = agg.fit_temperature(val.logits, y_enc[va], val.temperature)
            self.aggregator_ = "mean"
            self.agg_params_ = {}

    def predict_proba(self, X):
        check_is_fitted(self, "context_")
        out = self._outputs(X)
        return agg.aggregate_clf(self.aggregator_, out.logits, out.temperature * self.temperature_factor_, **self.agg_params_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TabPFNLabRegressor(RegressorMixin, _TabPFNLabBase):
    _task = "regression"
    _default_aggregator = "mixture"

    def _fit_predict(self, Xc, yc, X, n_estimators, seed):
        return self.backend_.reg_outputs(Xc, yc, X, n_estimators, seed)

    def fit(self, X, y):
        X_num = self._fit_common(X, y)
        agg.get_reg_aggregator(self.aggregator_)
        self._select_context(X, X_num, np.asarray(y, dtype=float))
        return self

    def predict_distribution(self, X, levels=agg.DEFAULT_LEVELS):
        check_is_fitted(self, "context_")
        out = self._outputs(X)
        return agg.aggregate_reg(self.aggregator_, out.probs, out.dist, levels, **self.agg_params_)

    def predict(self, X):
        return self.predict_distribution(X).mean
