import copy
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from tabpfn_lab.backends import SklearnBackend
from tabpfn_lab.cache import OutputCache
from tabpfn_lab.datasets import make_split
from tabpfn_lab.sizes import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


def tabpfn_available() -> bool:
    if os.environ.get("RUN_TABPFN", "1") == "0":
        return False
    try:
        import tabpfn  # noqa: F401
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    has_tabpfn = tabpfn_available()
    large = os.environ.get("RUN_TIER_LARGE") == "1"
    skip_tabpfn = pytest.mark.skip(reason="needs tabpfn + CUDA (set RUN_TABPFN=1 on the GPU server)")
    skip_large = pytest.mark.skip(reason="tier_large is opt-in: RUN_TIER_LARGE=1 on the GPU server")
    for item in items:
        if "tabpfn" in item.keywords and not has_tabpfn:
            item.add_marker(skip_tabpfn)
        if "tier_large" in item.keywords and not (large and has_tabpfn):
            item.add_marker(skip_large)


@pytest.fixture
def clf_split():
    X, y = make_synthetic(400, 6, n_classes=3, seed=0)
    return make_split(X, y, "classification", test_size=0.3, seed=0, name="clf")


@pytest.fixture
def reg_split():
    X, y = make_synthetic(400, 5, task="regression", seed=0)
    return make_split(X, y, "regression", test_size=0.3, seed=0, name="reg")


@pytest.fixture
def backend():
    return SklearnBackend(n_trees=10)


@pytest.fixture
def cache(tmp_path):
    return OutputCache(tmp_path / "cache")


def make_logits(E=8, n=50, C=3, seed=0, scale=2.0):
    """Per-estimator logits with known structure: a shared signal plus estimator-specific noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, scale, size=(1, n, C))
    return base + rng.normal(0, 0.5, size=(E, n, C))


@pytest.fixture
def logits_factory():
    return make_logits


def load_smoke(tmp_path, **overrides):
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["cache_dir"] = str(tmp_path / "raw")
    cfg["log_dir"] = str(tmp_path / "logs")
    cfg.update(overrides)
    cfg.setdefault("out_csv", str(tmp_path / "tables" / f"{cfg['name']}.csv"))
    return cfg


@pytest.fixture
def smoke(tmp_path):
    return lambda **kw: load_smoke(tmp_path, **kw)


class SpyBackend(SklearnBackend):
    """Sklearn backend that counts calls and checks shapes, class ordering, and that y_test never arrives."""

    def __init__(self, forbidden_y=None, **kw):
        super().__init__(**kw)
        self.calls = []
        self.forbidden_y = [] if forbidden_y is None else list(forbidden_y)

    def _check(self, name, X_train, y_train, X_test, classes=None):
        X_train, y_train, X_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test)
        assert X_train.ndim == 2 and X_test.ndim == 2 and X_train.shape[1] == X_test.shape[1]
        assert len(X_train) == len(y_train)
        if classes is not None:
            assert np.array_equal(np.asarray(classes), np.arange(len(classes))), "classes must be ordered codes"
        for fy in self.forbidden_y:
            assert not (len(fy) == len(y_train) and np.array_equal(fy, y_train)), "y_test reached the backend"
        self.calls.append((name, len(X_train), len(X_test)))

    def clf_outputs(self, X_train, y_train, X_test, classes, n_estimators, seed, overrides=None):
        self._check("clf_outputs", X_train, y_train, X_test, classes)
        out = super().clf_outputs(X_train, y_train, X_test, classes, n_estimators, seed, overrides)
        assert out.logits.shape == (n_estimators, len(X_test), len(classes))
        return out

    def reg_outputs(self, X_train, y_train, X_test, n_estimators, seed, overrides=None):
        self._check("reg_outputs", X_train, y_train, X_test)
        return super().reg_outputs(X_train, y_train, X_test, n_estimators, seed, overrides)

    def native_clf(self, X_train, y_train, X_test, classes, n_estimators, seed, tuned=False):
        self._check("native_clf", X_train, y_train, X_test, classes)
        return super().native_clf(X_train, y_train, X_test, classes, n_estimators, seed, tuned)

    def native_reg(self, X_train, y_train, X_test, n_estimators, seed, levels):
        self._check("native_reg", X_train, y_train, X_test)
        return super().native_reg(X_train, y_train, X_test, n_estimators, seed, levels)
