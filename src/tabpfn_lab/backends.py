"""Model backends. This is the only module allowed to import `tabpfn` (and `torch`).

A backend turns (train, test) arrays into raw per-estimator outputs:
classification -> ClfOutputs (E, n, C) raw logits plus the temperature the package applies;
regression -> RegOutputs (E, n, B) per-estimator probabilities on shared bins plus a BinDistribution.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("tabpfn_lab")

# logit given to classes the model never saw (e.g. a coreset that missed a class): probability ~ 0
MISSING_LOGIT = -50.0
# HalfNormal(1).icdf(0.5) == Phi^-1(0.75); used by tabpfn's FullSupportBarDistribution tails
_HN_Q50 = 0.6744897501960817
SUBSAMPLE_METHODS_FALLBACK = ["auto", "balanced", "stratified", "majority_downsample"]


# --------------------------------------------------------------------------- math helpers


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def align_logits(logits, model_classes, classes, fill=MISSING_LOGIT):
    """Place (..., C_model) logits into the full `classes` ordering; unseen classes get `fill`."""
    logits = np.asarray(logits)
    model_classes = list(np.asarray(model_classes).tolist())
    classes = list(np.asarray(classes).tolist())
    out = np.full(logits.shape[:-1] + (len(classes),), fill, dtype=np.float64)
    for j, c in enumerate(model_classes):
        out[..., classes.index(c)] = logits[..., j]
    return out


def align_proba(proba, model_classes, classes):
    """Place (..., C_model) probabilities into the full `classes` ordering; unseen classes get 0."""
    return align_logits(proba, model_classes, classes, fill=0.0)


# --------------------------------------------------------------------------- bin distributions


class BarDistribution:
    """Numpy port of tabpfn's (FullSupport)BarDistribution decoding on fixed raw-unit borders.

    All methods take log-probabilities (any unnormalised logits work, they are softmaxed) of shape (..., B).
    With `full_support=True` the outer buckets are half-normal tails, as in tabpfn's
    FullSupportBarDistribution (affects `mean` and `nll`; `icdf` is the plain piecewise-linear one, as there).
    """

    kind = "bar"

    def __init__(self, borders, full_support: bool = False):
        self.borders = np.asarray(borders, dtype=np.float64)
        self.widths = np.diff(self.borders)
        self.full_support = bool(full_support)
        if self.full_support:
            self.kind = "fullsupport"
            self._s0 = self.widths[0] / _HN_Q50
            self._s1 = self.widths[-1] / _HN_Q50
        else:
            self.kind = "histogram"

    @property
    def n_bins(self) -> int:
        return len(self.widths)

    def bucket_means(self) -> np.ndarray:
        means = self.borders[:-1] + self.widths / 2
        if self.full_support:
            means = means.copy()
            means[0] = self.borders[1] - self._s0 * np.sqrt(2 / np.pi)
            means[-1] = self.borders[-2] + self._s1 * np.sqrt(2 / np.pi)
        return means

    def mean(self, log_probs) -> np.ndarray:
        return softmax(log_probs) @ self.bucket_means()

    def icdf(self, log_probs, q: float) -> np.ndarray:
        p = softmax(log_probs)
        cum = np.cumsum(p, axis=-1)
        idx = np.clip((cum < q).sum(axis=-1), 0, self.n_bins - 1)
        cum0 = np.concatenate([np.zeros(cum.shape[:-1] + (1,)), cum], axis=-1)
        rest = q - np.take_along_axis(cum0, idx[..., None], -1)[..., 0]
        pi = np.take_along_axis(p, idx[..., None], -1)[..., 0]
        left = self.borders[idx]
        width = self.widths[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(pi > 0, rest / pi, 0.0)
        return left + width * frac

    def quantiles(self, log_probs, levels) -> np.ndarray:
        """(..., L) quantiles for the given levels."""
        return np.stack([self.icdf(log_probs, float(q)) for q in levels], axis=-1)

    def log_density(self, log_probs, y) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        lp = log_softmax(log_probs)
        idx = np.searchsorted(self.borders, y, side="left") - 1
        idx[y == self.borders[0]] = 0
        idx[y == self.borders[-1]] = self.n_bins - 1
        idx = np.clip(idx, 0, self.n_bins - 1)
        out = np.take_along_axis(lp, idx[..., None], -1)[..., 0] - np.log(self.widths[idx])
        if self.full_support:
            first, last = idx == 0, idx == self.n_bins - 1
            out[first] += _halfnormal_logpdf(np.maximum(self.borders[1] - y[first], 1e-8), self._s0) + np.log(self.widths[0])
            out[last] += _halfnormal_logpdf(np.maximum(y[last] - self.borders[-2], 1e-8), self._s1) + np.log(self.widths[-1])
        return out

    def nll(self, log_probs, y) -> float:
        return float(-np.mean(self.log_density(log_probs, y)))


def _halfnormal_logpdf(x, s):
    return np.log(2.0) - 0.5 * np.log(2 * np.pi) - np.log(s) - x**2 / (2 * s**2)


def make_distribution(kind: str, borders) -> BarDistribution:
    if kind == "histogram":
        return BarDistribution(borders, full_support=False)
    if kind == "fullsupport":
        return BarDistribution(borders, full_support=True)
    raise ValueError(f"unknown distribution kind {kind!r}")


# --------------------------------------------------------------------------- output containers


@dataclass
class ClfOutputs:
    logits: np.ndarray  # (E, n_test, C) raw logits, columns ordered like `classes`
    temperature: float  # per-estimator softmax temperature the package would apply
    classes: np.ndarray
    meta: dict = field(default_factory=dict)

    def subset(self, k: int) -> "ClfOutputs":
        return ClfOutputs(self.logits[:k], self.temperature, self.classes, dict(self.meta))

    def to_arrays(self) -> dict:
        return {"logits": self.logits, "temperature": np.float64(self.temperature), "classes": np.asarray(self.classes)}

    @classmethod
    def from_arrays(cls, arrays: dict, meta=None) -> "ClfOutputs":
        return cls(arrays["logits"], float(arrays["temperature"]), arrays["classes"], meta or {})


@dataclass
class RegOutputs:
    probs: np.ndarray  # (E, n_test, B) per-estimator probabilities on shared bins
    dist: BarDistribution  # decodes aggregated log-probs in raw target units
    y_mean: float
    y_std: float
    meta: dict = field(default_factory=dict)

    def subset(self, k: int) -> "RegOutputs":
        return RegOutputs(self.probs[:k], self.dist, self.y_mean, self.y_std, dict(self.meta))

    def to_arrays(self) -> dict:
        return {
            "probs": self.probs,
            "borders": self.dist.borders,
            "dist_kind": np.array(self.dist.kind),
            "y_mean": np.float64(self.y_mean),
            "y_std": np.float64(self.y_std),
        }

    @classmethod
    def from_arrays(cls, arrays: dict, meta=None) -> "RegOutputs":
        dist = make_distribution(str(arrays["dist_kind"]), arrays["borders"])
        return cls(arrays["probs"], dist, float(arrays["y_mean"]), float(arrays["y_std"]), meta or {})


# --------------------------------------------------------------------------- backend interface


class Backend(ABC):
    name: str = "backend"
    accepts_raw = False  # True if the backend can take DataFrames / strings directly

    @abstractmethod
    def params(self) -> dict:
        """JSON-able description used in cache keys."""

    @abstractmethod
    def clf_outputs(self, X_train, y_train, X_test, classes, n_estimators, seed, overrides=None) -> ClfOutputs: ...

    @abstractmethod
    def reg_outputs(self, X_train, y_train, X_test, n_estimators, seed, overrides=None) -> RegOutputs: ...

    @abstractmethod
    def native_clf(self, X_train, y_train, X_test, classes, n_estimators, seed, tuned=False) -> np.ndarray:
        """(n, C) probabilities from the package's own prediction path."""

    @abstractmethod
    def native_reg(self, X_train, y_train, X_test, n_estimators, seed, levels) -> dict:
        """{'mean': (n,), 'median': (n,), 'quantiles': (n, L)} from the package's own prediction path."""

    @abstractmethod
    def embed(self, X_fit, y_fit, X, seed, task, n_estimators=1) -> np.ndarray:
        """(E, n, dim) embeddings of rows X after fitting on (X_fit, y_fit)."""

    def subsample_methods(self) -> list[str]:
        return list(SUBSAMPLE_METHODS_FALLBACK)


# --------------------------------------------------------------------------- built-in subsampling emulation


def emulate_subsample(y, k, method, rng, task):
    """Row indices for one estimator, emulating tabpfn's SUBSAMPLE_SAMPLES / SAMPLE_SUBSAMPLING_METHOD."""
    from .coresets import allocate  # local import: coresets imports this module

    y = np.asarray(y)
    n = len(y)
    k = int(k)
    if method == "auto":
        method = "stratified" if task == "classification" else "balanced"
    if method == "majority_downsample":
        values, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        top = counts.max()
        if (counts == top).sum() != 1:
            warnings.warn("majority_downsample: no unique majority value; falling back", stacklevel=2)
            method = "stratified" if task == "classification" else "balanced"
        else:
            maj = int(np.argmax(counts))
            non_maj = np.flatnonzero(inverse != maj)
            if k <= len(non_maj):
                raise ValueError(
                    f"SUBSAMPLE_SAMPLES ({k}) must exceed the number of non-majority rows ({len(non_maj)}) "
                    "when using majority_downsample."
                )
            maj_rows = np.flatnonzero(inverse == maj)
            take = rng.choice(maj_rows, min(k - len(non_maj), len(maj_rows)), replace=False)
            return np.sort(np.concatenate([non_maj, take]))
    if k >= n:
        return np.arange(n)
    if method == "stratified" and task == "classification":
        groups = [np.flatnonzero(y == c) for c in np.unique(y)]
        alloc = allocate(np.array([len(g) for g in groups]), k, min_one=True)
        return np.sort(np.concatenate([rng.choice(g, a, replace=False) for g, a in zip(groups, alloc)]))
    # "balanced" (label-agnostic) or anything else: uniform without replacement
    return np.sort(rng.choice(n, k, replace=False))


# --------------------------------------------------------------------------- sklearn backend


class SklearnBackend(Backend):
    """Cheap random-forest stand-in used by the unit tests. Estimator k is a RandomForest with random_state = seed*1000 + k."""

    name = "sklearn"

    def __init__(self, n_trees: int = 10, n_bins: int = 40, eps: float = 1e-3, alpha: float = 0.01, max_depth=None):
        self.n_trees = int(n_trees)
        self.n_bins = int(n_bins)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.max_depth = max_depth

    def params(self) -> dict:
        return {"name": self.name, "n_trees": self.n_trees, "n_bins": self.n_bins, "eps": self.eps, "alpha": self.alpha, "max_depth": self.max_depth}

    def _forest(self, task, rs):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(n_estimators=self.n_trees, random_state=int(rs), n_jobs=1, max_depth=self.max_depth)

    def _subsets(self, y_train, n_estimators, seed, overrides, task):
        if not overrides or overrides.get("SUBSAMPLE_SAMPLES") is None:
            return [None] * n_estimators
        k = overrides["SUBSAMPLE_SAMPLES"]
        method = overrides.get("SAMPLE_SUBSAMPLING_METHOD", "auto")
        if method not in self.subsample_methods():
            raise ValueError(f"unknown SAMPLE_SUBSAMPLING_METHOD {method!r}")
        out = []
        for e in range(n_estimators):
            rng = np.random.default_rng([int(seed), e, 7])
            out.append(emulate_subsample(y_train, k, method, rng, task))
        return out

    def clf_outputs(self, X_train, y_train, X_test, classes, n_estimators, seed, overrides=None) -> ClfOutputs:
        X_train, X_test, y_train = np.asarray(X_train, float), np.asarray(X_test, float), np.asarray(y_train)
        logits = []
        for e, rows in enumerate(self._subsets(y_train, n_estimators, seed, overrides, "classification")):
            Xe, ye = (X_train, y_train) if rows is None else (X_train[rows], y_train[rows])
            rf = self._forest("classification", seed * 1000 + e).fit(Xe, ye)
            p = rf.predict_proba(X_test)
            c = p.shape[1]
            p = p * (1 - c * self.eps) + self.eps
            logits.append(align_logits(np.log(p), rf.classes_, classes))
        meta = {"backend": self.name, "n_estimators_": int(n_estimators), "params": self.params()}
        return ClfOutputs(np.stack(logits), 1.0, np.asarray(classes), meta)

    def _borders(self, y_train):
        y = np.asarray(y_train, float)
        b = np.unique(np.quantile(y, np.linspace(0, 1, self.n_bins + 1)))
        if len(b) < 3:
            b = np.array([y.min() - 1.0, y.min(), y.min() + 1.0])
        margin = 0.1 * (b[-1] - b[0])
        b = b.copy()
        b[0] -= margin
        b[-1] += margin
        return b

    def reg_outputs(self, X_train, y_train, X_test, n_estimators, seed, overrides=None) -> RegOutputs:
        X_train, X_test, y_train = np.asarray(X_train, float), np.asarray(X_test, float), np.asarray(y_train, float)
        borders = self._borders(y_train)
        B = len(borders) - 1
        n = len(X_test)
        probs = []
        for e, rows in enumerate(self._subsets(y_train, n_estimators, seed, overrides, "regression")):
            Xe, ye = (X_train, y_train) if rows is None else (X_train[rows], y_train[rows])
            rf = self._forest("regression", seed * 1000 + e).fit(Xe, ye)
            preds = np.stack([t.predict(X_test) for t in rf.estimators_], axis=1)  # (n, trees)
            idx = np.clip(np.searchsorted(borders, preds, side="right") - 1, 0, B - 1)
            counts = np.zeros((n, B))
            np.add.at(counts, (np.repeat(np.arange(n), preds.shape[1]), idx.ravel()), 1.0)
            probs.append((counts + self.alpha) / (preds.shape[1] + self.alpha * B))
        meta = {"backend": self.name, "n_estimators_": int(n_estimators), "params": self.params()}
        y_mean, y_std = float(y_train.mean()), float(y_train.std() or 1.0)
        return RegOutputs(np.stack(probs), BarDistribution(borders, full_support=False), y_mean, y_std, meta)

    def native_clf(self, X_train, y_train, X_test, classes, n_estimators, seed, tuned=False) -> np.ndarray:
        from .aggregation import aggregate_clf, fit_temperature

        out = self.clf_outputs(X_train, y_train, X_test, classes, n_estimators, seed)
        if not tuned:
            return aggregate_clf("mean", out.logits, out.temperature)
        # emulate the package's temperature calibration on an internal holdout of the training data
        from sklearn.model_selection import train_test_split

        y_train = np.asarray(y_train)
        strat = y_train if np.bincount(y_train.astype(int)).min() >= 2 else None
        tr, ho = train_test_split(np.arange(len(y_train)), test_size=0.2, random_state=seed, stratify=strat)
        X_train = np.asarray(X_train, float)
        val = self.clf_outputs(X_train[tr], y_train[tr], X_train[ho], classes, n_estimators, seed)
        factor = fit_temperature(val.logits, y_train[ho], val.temperature)
        return aggregate_clf("mean", out.logits, out.temperature * factor)

    def native_reg(self, X_train, y_train, X_test, n_estimators, seed, levels) -> dict:
        from .aggregation import aggregate_reg

        out = self.reg_outputs(X_train, y_train, X_test, n_estimators, seed)
        pred = aggregate_reg("mixture", out.probs, out.dist, levels)
        return {"mean": pred.mean, "median": out.dist.icdf(pred.logp, 0.5), "quantiles": pred.quantiles}

    def embed(self, X_fit, y_fit, X, seed, task, n_estimators=1) -> np.ndarray:
        """Per-estimator embedding = per-tree predictions of a forest fit on (X_fit, y_fit)."""
        X_fit, X = np.asarray(X_fit, float), np.asarray(X, float)
        embs = []
        for e in range(n_estimators):
            rf = self._forest(task, seed * 1000 + e).fit(X_fit, y_fit)
            if task == "classification":
                embs.append(np.concatenate([t.predict_proba(X) for t in rf.estimators_], axis=1))
            else:
                embs.append(np.stack([t.predict(X) for t in rf.estimators_], axis=1))
        return np.stack(embs)


# --------------------------------------------------------------------------- tabpfn backend

_VERSION_MAP = {
    "2": "V2",
    "2.5": "V2_5",
    "2.6": "V2_6",
    "3": "V3",
    "3.5": "V3_5",
    "3.5-fast": "V3_5_FAST",
}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _config_to_dict(cfg) -> dict:
    try:
        return _jsonable(dataclasses.asdict(cfg))
    except TypeError:
        return _jsonable(dict(vars(cfg)))


def is_oom_error(exc: BaseException) -> bool:
    """True for torch CUDA/MPS OOM and tabpfn's own OOM wrappers (tabpfn.errors.TabPFNOutOfMemoryError)."""
    names = {c.__name__ for c in type(exc).__mro__}
    if names & {"OutOfMemoryError", "TabPFNOutOfMemoryError"}:
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class TabPFNBackend(Backend):
    """Real TabPFN. Per-estimator classification logits come from `predict_raw_logits`; per-estimator
    regression distributions are captured by replicating `TabPFNRegressor._compute_aggregated_logits`
    (tabpfn==9.0.0; guarded by tests/test_tabpfn_integration.py)."""

    accepts_raw = True

    def __init__(self, version="3.5", device="cuda", fit_mode="fit_preprocessors", inference_precision="auto", **extra):
        if str(version) not in _VERSION_MAP:
            raise ValueError(f"unknown TabPFN version {version!r}; choose from {sorted(_VERSION_MAP)}")
        self.version = str(version)
        self.device = device
        self.fit_mode = fit_mode
        self.inference_precision = inference_precision
        self.extra = extra
        self.name = f"tabpfn-{self.version}"
        self._config_dumped: dict[str, dict] = {}

    def params(self) -> dict:
        return _jsonable(
            {
                "name": self.name,
                "version": self.version,
                "fit_mode": self.fit_mode,
                "inference_precision": str(self.inference_precision),
                **self.extra,
            }
        )

    def _precision(self):
        p = self.inference_precision
        if isinstance(p, str) and p not in ("auto", "autocast"):
            import torch

            return getattr(torch, p)
        return p

    def _make(self, task, n_estimators, seed, **kw):
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.constants import ModelVersion
        except ImportError as e:  # pragma: no cover - exercised on machines without tabpfn
            raise ImportError("TabPFNBackend needs `pip install -e '.[tabpfn]'` (tabpfn==9.0.0)") from e
        cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
        kwargs = dict(
            n_estimators=int(n_estimators),
            random_state=int(seed),
            device=self.device,
            fit_mode=self.fit_mode,
            inference_precision=self._precision(),
        )
        kwargs.update(self.extra)
        kwargs.update({k: v for k, v in kw.items() if v is not None})
        return cls.create_default_for_version(getattr(ModelVersion, _VERSION_MAP[self.version]), **kwargs)

    def _fit(self, model, X, y):
        try:
            return model.fit(X, y)
        except Exception as e:
            if type(e).__name__ in ("TabPFNLicenseError", "TabPFNHuggingFaceGatedRepoError"):
                raise RuntimeError(
                    "TabPFN weights could not be downloaded: accept the licence and set TABPFN_TOKEN "
                    "(and HF_TOKEN for gated Hugging Face repos). See README 'GPU server setup'.\n" + str(e)
                ) from e
            raise

    def _meta(self, model, task) -> dict:
        import tabpfn
        from tabpfn.base import resolved_softmax_temperature

        meta = {
            "backend": self.name,
            "tabpfn_version": tabpfn.__version__,
            "model_version": self.version,
            "n_estimators_": int(getattr(model, "n_estimators_", -1)),
            "device": str(self.device),
            "inference_precision": str(self.inference_precision),
            "softmax_temperature": float(resolved_softmax_temperature(model)),
        }
        if task not in self._config_dumped:
            self._config_dumped[task] = _config_to_dict(model.get_inference_config())
        meta["inference_config"] = self._config_dumped[task]
        return meta

    def clf_outputs(self, X_train, y_train, X_test, classes, n_estimators, seed, overrides=None) -> ClfOutputs:
        from tabpfn.base import resolved_softmax_temperature

        clf = self._make("classification", n_estimators, seed, inference_config=overrides)
        self._fit(clf, X_train, y_train)
        raw = clf.predict_raw_logits(X_test)  # (E, n, C_model), class permutations undone, no temperature
        logits = align_logits(raw, clf.classes_, classes)
        return ClfOutputs(logits, float(resolved_softmax_temperature(clf)), np.asarray(classes), self._meta(clf, "classification"))

    def native_clf(self, X_train, y_train, X_test, classes, n_estimators, seed, tuned=False) -> np.ndarray:
        kw = {"tuning_config": {"calibrate_temperature": True}} if tuned else {}
        clf = self._make("classification", n_estimators, seed, **kw)
        self._fit(clf, X_train, y_train)
        return align_proba(clf.predict_proba(X_test), clf.classes_, classes)

    def _capture_reg(self, reg, X_test) -> np.ndarray:
        """Per-estimator probabilities on znorm_space_bardist_ borders: the loop of
        TabPFNRegressor._compute_aggregated_logits without the averaging."""
        import torch
        from sklearn import config_context
        from tabpfn.preprocessing.clean import clean_data_transform
        from tabpfn.preprocessing.datamodel import FeatureModality
        from tabpfn.utils import translate_probs_across_borders
        from tabpfn.validation import check_input_shape_matches, ensure_compatible_predict_input_sklearn

        with config_context(transform_output="default"):
            X = X_test
            check_input_shape_matches(X, estimator=reg)
            X = reg.date_transformer_.transform(X)
            X = reg.text_transformer_.transform(X)
            X = ensure_compatible_predict_input_sklearn(X, reg)
            X = clean_data_transform(
                X,
                cat_indices=reg.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL),
                ord_encoder=getattr(reg, "ordinal_encoder_", None),
                passthrough_inf=reg.get_inference_config().PASSTHROUGH_INF,
            )
            probs = []
            to = reg.znorm_space_bardist_.borders
            # _iter_forward_executor already applies the per-estimator softmax temperature and
            # target-transform border mapping; translate_probs_across_borders takes logits.
            for borders_t, output in reg._iter_forward_executor(X, use_inference_mode=True):
                p = translate_probs_across_borders(
                    output,
                    frm=torch.as_tensor(borders_t, device=output.device),
                    to=to.to(output.device),
                )
                probs.append(p.float().cpu().numpy())
        return np.stack(probs).astype(np.float64)

    def reg_outputs(self, X_train, y_train, X_test, n_estimators, seed, overrides=None) -> RegOutputs:
        reg = self._make("regression", n_estimators, seed, inference_config=overrides)
        self._fit(reg, X_train, y_train)
        if getattr(reg, "is_constant_target_", False):
            raise ValueError("constant regression target: no per-estimator distributions to capture")
        probs = self._capture_reg(reg, X_test)
        borders = reg.raw_space_bardist_.borders.detach().cpu().numpy().astype(np.float64)
        meta = self._meta(reg, "regression")
        meta["ensemble_softmax_temperature"] = float(getattr(reg, "ensemble_softmax_temperature_", 1.0))
        return RegOutputs(probs, BarDistribution(borders, full_support=True), float(reg.y_train_mean_), float(reg.y_train_std_), meta)

    def native_reg(self, X_train, y_train, X_test, n_estimators, seed, levels) -> dict:
        reg = self._make("regression", n_estimators, seed)
        self._fit(reg, X_train, y_train)
        out = reg.predict(X_test, output_type="main", quantiles=[float(q) for q in levels])
        return {"mean": np.asarray(out["mean"]), "median": np.asarray(out["median"]), "quantiles": np.stack(out["quantiles"], axis=1)}

    def embed(self, X_fit, y_fit, X, seed, task, n_estimators=1, chunk=4096) -> np.ndarray:
        model = self._make(task, n_estimators, seed)
        self._fit(model, X_fit, y_fit)
        parts = [model.get_embeddings(X[i : i + chunk], data_source="test") for i in range(0, len(X), chunk)]
        return np.concatenate([np.asarray(p) if np.ndim(p) == 3 else np.asarray(p)[None] for p in parts], axis=1)

    def subsample_methods(self) -> list[str]:
        from tabpfn.preprocessing.configs import SampleSubsamplingMethod

        return [m.value for m in SampleSubsamplingMethod]


def make_backend(spec: dict | None) -> Backend:
    """Build a backend from a config dict like {name: tabpfn, version: "3.5", device: cuda} or {name: sklearn}."""
    spec = dict(spec or {"name": "sklearn"})
    name = spec.pop("name", "sklearn")
    if name == "sklearn":
        return SklearnBackend(**spec)
    if name == "tabpfn":
        return TabPFNBackend(**spec)
    raise ValueError(f"unknown backend {name!r}")


# --------------------------------------------------------------------------- helpers used by the runner


def translate_probs(probs, frm, to) -> np.ndarray:
    """Re-bin (..., B_frm) probabilities from borders `frm` onto borders `to` (piecewise-uniform CDF), like
    tabpfn.utils.translate_probs_across_borders but taking probabilities."""
    probs = np.asarray(probs, np.float64)
    frm = np.asarray(frm, np.float64)
    to = np.asarray(to, np.float64)
    cdf_frm = np.concatenate([np.zeros(probs.shape[:-1] + (1,)), np.cumsum(probs, axis=-1)], axis=-1)
    flat = cdf_frm.reshape(-1, cdf_frm.shape[-1])
    cdf_to = np.stack([np.interp(to, frm, row) for row in flat]).reshape(probs.shape[:-1] + (len(to),))
    cdf_to[..., 0] = 0.0
    cdf_to[..., -1] = 1.0
    return np.clip(np.diff(cdf_to, axis=-1), 0.0, None)


def reset_gpu_peak() -> None:
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    """Peak CUDA memory since the last reset, or NaN without torch/CUDA (never imports torch itself)."""
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    return float("nan")
