import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from tabpfn_lab import aggregation as A
from tabpfn_lab import coresets as cs
from tabpfn_lab.backends import SklearnBackend
from tabpfn_lab.estimator import TabPFNLabClassifier, TabPFNLabRegressor
from tabpfn_lab.sizes import make_synthetic


def clf_data(n=200):
    X, y = make_synthetic(n, 4, n_classes=3, seed=0)
    return X, y


def make_clf(**kw):
    return TabPFNLabClassifier(backend="sklearn", n_estimators=3, **kw)


def test_sklearn_contract_classifier():
    X, y = clf_data()
    est = make_clf(coreset="stratified", budget=60)
    assert clone(est).get_params() == est.get_params()
    est.set_params(n_estimators=2)
    assert est.n_estimators == 2
    with pytest.raises(Exception):
        check_is_fitted(est)
    est.fit(X, y)
    check_is_fitted(est)
    assert est.predict(X[:10]).shape == (10,)
    df = pd.DataFrame(X, columns=list("abcd"))
    df["cat"] = np.where(X[:, 0] > 0, "hi", "lo")
    df.loc[3, "a"] = np.nan
    p = make_clf().fit(df, y).predict_proba(df.iloc[:7])
    assert p.shape == (7, 3)
    np.testing.assert_allclose(p.sum(1), 1)


def test_sklearn_contract_regressor():
    X, y = make_synthetic(200, 4, task="regression", seed=1)
    est = TabPFNLabRegressor(backend="sklearn", n_estimators=2, coreset="random", budget=80)
    assert clone(est).get_params() == est.get_params()
    est.fit(pd.DataFrame(X), y)
    pred = est.predict(pd.DataFrame(X[:5]))
    assert pred.shape == (5,) and np.isfinite(pred).all()
    for agg in A.REG_AGGREGATORS:
        assert np.isfinite(TabPFNLabRegressor(backend="sklearn", n_estimators=2, aggregator=agg).fit(X, y).predict(X[:5])).all()


def test_reproduces_backend_mean_path_when_default():
    X, y = clf_data()
    be = SklearnBackend()
    est = make_clf(aggregator="mean", coreset="none").fit(X[:150], y[:150])
    ref = be.native_clf(X[:150], y[:150], X[150:], np.arange(3), 3, 0)
    np.testing.assert_allclose(est.predict_proba(X[150:]), ref, atol=1e-12)


@pytest.mark.tabpfn
def test_reproduces_stock_tabpfn_when_default():
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    X, y = clf_data(300)
    est = TabPFNLabClassifier(n_estimators=4, random_state=0, device="cuda").fit(X[:200], y[:200])
    stock = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, n_estimators=4, random_state=0, device="cuda").fit(X[:200], y[:200])
    np.testing.assert_allclose(est.predict_proba(X[200:]), stock.predict_proba(X[200:]), atol=1e-4)


def test_coreset_selected_in_fit_only(monkeypatch):
    X, y = clf_data()
    seen = []
    real = cs.select_coreset

    def spy(method, Xs, ys, budget, seed, task="classification", **kw):
        seen.append(len(Xs))
        return real(method, Xs, ys, budget, seed, task, **kw)

    monkeypatch.setattr(cs, "select_coreset", spy)
    est = make_clf(coreset="kmeans_stratified", budget=50).fit(X[:150], y[:150])
    est.predict_proba(X[150:])
    est.predict(X[:20])
    assert seen == [150]


@pytest.mark.parametrize("aggregator", ["val_weighted", "mean_temp_offline"])
def test_validation_slice_from_train_only(aggregator):
    X, y = clf_data()
    est = make_clf(aggregator=aggregator).fit(X[:150], y[:150])
    params = (dict(est.agg_params_), est.temperature_factor_)
    poisoned = X[150:] * 1000 + 50
    est.predict_proba(poisoned)
    after = (dict(est.agg_params_), est.temperature_factor_)
    assert params[1] == after[1]
    for k in params[0]:
        np.testing.assert_array_equal(params[0][k], after[0][k])


def test_string_and_noncontiguous_labels():
    X, y = clf_data()
    labels = np.array(["zeta", "alpha", "mid"])[y]
    est = make_clf().fit(X, labels)
    assert list(est.classes_) == ["alpha", "mid", "zeta"]
    assert set(est.predict(X)) <= set(labels)
    codes = np.array([3, 10, 7])[y]
    est2 = make_clf().fit(X, codes)
    assert list(est2.classes_) == [3, 7, 10]
    P = est2.predict_proba(X)
    np.testing.assert_array_equal(est2.classes_[P.argmax(1)], est2.predict(X))


def test_budget_fraction_resolved_at_fit():
    X, y = clf_data()
    est = make_clf(coreset="random", budget_fraction=0.1).fit(X[:150], y[:150])
    assert est.budget_ == 15 and len(est.context_[0][1]) == 15
    est.fit(X, y)
    assert est.budget_ == 20
    full = make_clf(coreset="random", budget=500).fit(X, y)
    assert full.budget_ == len(X) and len(full.context_[0][1]) == len(X)


def test_deterministic_picklable_clonable_cv():
    X, y = clf_data(120)
    a = make_clf(coreset="stratified", budget=40, random_state=3).fit(X, y).predict_proba(X)
    b = make_clf(coreset="stratified", budget=40, random_state=3).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    est = make_clf(coreset="stratified", budget=40).fit(X, y)
    est2 = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(est.predict_proba(X), est2.predict_proba(X))
    scores = cross_val_score(make_clf(coreset="perest_partition", budget=40), X, y, cv=3)
    assert len(scores) == 3 and np.all(scores > 0.3)


def test_no_benchmark_code_in_core():
    code = (
        "import sys, tabpfn_lab, tabpfn_lab.estimator, tabpfn_lab.experiments, tabpfn_lab.cli\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('tabarena', 'autogluon', 'talent', 'beyondarena', 'benchmarks', 'tabpfn', 'torch')]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == ""
