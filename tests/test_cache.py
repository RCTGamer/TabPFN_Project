import numpy as np
import pytest

from tabpfn_lab.cache import OutputCache
from tabpfn_lab.experiments import run_aggregation

from conftest import SpyBackend

BASE = dict(dataset="d", task="classification", split_seed=0, n_train=100, n_test=50, backend={"name": "sklearn", "n_trees": 10}, n_estimators=8, subset_id="full")


def test_roundtrip_exact(cache):
    arrays = {"a": np.random.default_rng(0).normal(size=(3, 4, 5)), "b": np.arange(5, dtype=np.int32), "t": np.float64(0.9)}
    cache.put("k1", arrays, {"x": 1})
    got, meta = cache.get("k1")
    for k, v in arrays.items():
        np.testing.assert_array_equal(got[k], v)
        assert got[k].dtype == np.asarray(v).dtype
    assert meta == {"x": 1}


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": "e"},
        {"split_seed": 1},
        {"seed": 7},
        {"n_estimators": 4},
        {"backend": {"name": "sklearn", "n_trees": 11}},
        {"subset_id": "random:50:0"},
        {"n_train": 101},
    ],
)
def test_key_sensitivity(change):
    assert OutputCache.make_key(**{**BASE, **change}) != OutputCache.make_key(**BASE)


def test_key_ignores_dict_order():
    a = OutputCache.make_key(**{**BASE, "backend": {"name": "x", "a": 1, "b": 2}})
    b = OutputCache.make_key(**{**BASE, "backend": {"b": 2, "a": 1, "name": "x"}})
    assert a == b


def test_hit_skips_backend(smoke, tmp_path):
    cfg = smoke(datasets=[{"name": "syn", "source": "synthetic", "task": "classification", "n": 300, "n_classes": 3}], seeds=[0])
    be = SpyBackend()
    run_aggregation(cfg, backend=be)
    assert be.calls
    be2 = SpyBackend()
    cfg2 = dict(cfg, out_csv=str(tmp_path / "second.csv"))  # fresh CSV so resume does not skip the cell
    run_aggregation(cfg2, backend=be2)
    assert be2.calls == []


def test_atomic_write(cache, monkeypatch):
    def boom(f, **kw):
        f.write(b"partial")
        raise OSError("disk died")

    monkeypatch.setattr(np, "savez", boom)
    with pytest.raises(OSError):
        cache.put("k2", {"a": np.zeros(3)})
    assert not cache.exists("k2")
    assert not list(cache.root.rglob("*.tmp"))


def test_cached_equals_recomputed(clf_split, backend, cache):
    s = clf_split
    out = backend.clf_outputs(s.X_train, s.y_train, s.X_test, s.classes, 4, 0)
    cache.put("k3", out.to_arrays(), out.meta)
    arrays, _ = cache.get("k3")
    fresh = backend.clf_outputs(s.X_train, s.y_train, s.X_test, s.classes, 4, 0)
    np.testing.assert_array_equal(arrays["logits"], fresh.logits)
