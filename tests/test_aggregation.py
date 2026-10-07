import numpy as np
import pytest
import yaml

from tabpfn_lab import aggregation as A
from tabpfn_lab.backends import BarDistribution, softmax
from tabpfn_lab.metrics import log_loss

from conftest import ROOT, make_logits

UNWEIGHTED = ["mean", "logit_mean", "median", "trimmed_mean", "entropy_weighted"]


def _val(E, C=3, seed=5, n=60):
    rng = np.random.default_rng(seed)
    return make_logits(E, n, C, seed=seed), rng.integers(0, C, n)


def _call(name, L, T=1.0, **kw):
    if name in A.NEEDS_VALIDATION:
        vl, vy = _val(L.shape[0], L.shape[2])
        kw.update(val_logits=vl, val_y=vy)
    if name in A.NEEDS_WEIGHTS:
        kw.setdefault("w", np.full(L.shape[0], 1.0 / L.shape[0]))
    return A.aggregate_clf(name, L, T, **kw)


@pytest.mark.parametrize("name", sorted(A.CLF_AGGREGATORS))
def test_valid_distribution(name):
    L = make_logits(8, 40, 4, seed=1)
    P = _call(name, L, T=0.9)
    assert P.shape == (40, 4)
    assert np.isfinite(P).all() and (P >= 0).all()
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", sorted(set(A.CLF_AGGREGATORS) - A.CHANGES_SINGLE_ESTIMATOR))
def test_single_estimator_identity(name):
    # mean_temp_offline refits the temperature, so by design it changes a single estimator's output
    L = make_logits(1, 30, 3, seed=2)
    np.testing.assert_allclose(_call(name, L, T=0.9), softmax(L[0] / 0.9), atol=1e-9)


@pytest.mark.parametrize("name", UNWEIGHTED)
def test_identical_estimators(name):
    L1 = make_logits(1, 30, 3, seed=3)
    L = np.repeat(L1, 6, axis=0)
    np.testing.assert_allclose(_call(name, L), softmax(L1[0]), atol=1e-9)


@pytest.mark.parametrize("name", sorted(set(A.CLF_AGGREGATORS)))
def test_estimator_permutation_invariance(name):
    L = make_logits(8, 30, 3, seed=4)
    perm = np.random.default_rng(0).permutation(8)
    kw = {}
    if name == "weighted":
        kw["w"] = np.full(8, 1 / 8)  # fixed uniform w keeps the rule permutation invariant
    if name in A.NEEDS_VALIDATION:
        vl, vy = _val(8)
        kw_a = dict(kw, val_logits=vl, val_y=vy)
        kw_b = dict(kw, val_logits=vl[perm], val_y=vy)
    else:
        kw_a = kw_b = kw
    np.testing.assert_allclose(A.aggregate_clf(name, L, 1.0, **kw_a), A.aggregate_clf(name, L[perm], 1.0, **kw_b), atol=1e-10)


def test_mean_bounds():
    L = make_logits(8, 30, 3, seed=5)
    P_e = softmax(L)
    P = A.agg_mean(L, 1.0)
    assert (P >= P_e.min(0) - 1e-12).all() and (P <= P_e.max(0) + 1e-12).all()


def test_median_robust_to_outlier():
    consensus = np.log(np.array([0.7, 0.2, 0.1]))
    L = np.tile(consensus, (8, 20, 1)) + np.random.default_rng(0).normal(0, 0.05, (8, 20, 3))
    L[7] = np.log(np.array([0.001, 0.001, 0.998]))  # adversarial estimator
    target = softmax(consensus)
    err = lambda P: np.abs(P - target).mean()  # noqa: E731
    assert err(A.agg_median(L, 1.0)) < err(A.agg_mean(L, 1.0))
    assert err(A.agg_trimmed_mean(L, 1.0)) < err(A.agg_mean(L, 1.0))


def test_logit_mean_vs_mean_on_disagreement():
    # two estimators, binary: logits (log 0.9, log 0.1) and (log 0.1, log 0.9) for one row
    L = np.log(np.array([[[0.9, 0.1]], [[0.2, 0.8]]]))
    # mean of probabilities: (0.9+0.2)/2 = 0.55
    np.testing.assert_allclose(A.agg_mean(L, 1.0), [[0.55, 0.45]])
    # logit mean: normalised geometric mean sqrt(.9*.2) : sqrt(.1*.8) = 0.42426 : 0.28284
    g = np.array([np.sqrt(0.18), np.sqrt(0.08)])
    np.testing.assert_allclose(A.agg_logit_mean(L, 1.0), [g / g.sum()], atol=1e-12)
    np.testing.assert_allclose(A.agg_logit_mean(L, 1.0)[0, 0], 0.6, atol=1e-12)


def test_entropy_weighted_prefers_confident():
    confident = np.log(np.array([[0.98, 0.01, 0.01]]))
    uniform = np.log(np.array([[0.34, 0.33, 0.33]]))
    L = np.stack([confident, uniform])
    P_low = A.agg_entropy_weighted(L, 1.0, tau=0.05)
    assert P_low[0, 0] > A.agg_mean(L, 1.0)[0, 0]
    assert abs(P_low[0, 0] - 0.98) < 0.01
    np.testing.assert_allclose(A.agg_entropy_weighted(L, 1.0, tau=1e6), A.agg_mean(L, 1.0), atol=1e-6)


def test_weighted_matches_mean_for_uniform_w():
    L = make_logits(5, 30, 3, seed=6)
    np.testing.assert_allclose(A.agg_weighted(L, 0.8, w=np.full(5, 0.2)), A.agg_mean(L, 0.8), atol=1e-12)


def test_val_weighted_uses_only_validation_inputs():
    L = make_logits(6, 30, 3, seed=7)
    vl, vy = _val(6)
    w = A.fit_val_weights(vl, vy, 1.0)
    P = A.agg_val_weighted(L, 1.0, val_logits=vl, val_y=vy)
    # weights do not depend on the test-side logits: poisoning them only changes the output through L itself
    poisoned = L.copy()
    poisoned[:, :, 0] += 100.0
    w2 = A.fit_val_weights(vl, vy, 1.0)
    np.testing.assert_array_equal(w, w2)
    np.testing.assert_allclose(P, A.agg_weighted(L, 1.0, w=w))
    np.testing.assert_allclose(A.agg_val_weighted(poisoned, 1.0, val_logits=vl, val_y=vy), A.agg_weighted(poisoned, 1.0, w=w))
    with pytest.raises(ValueError):
        A.agg_val_weighted(L, 1.0)


def test_mean_temp_offline_improves_or_matches_val_logloss():
    for seed in range(5):
        vl, vy = _val(4, seed=seed)
        vl = vl * 3.0  # overconfident
        s = A.fit_temperature(vl, vy, 1.0)
        assert log_loss(vy, A.agg_mean(vl, s)) <= log_loss(vy, A.agg_mean(vl, 1.0)) + 1e-12


def test_logit_mean_equals_softmax_of_mean_logits():
    L = np.random.default_rng(8).normal(0, 3, (7, 25, 5))
    np.testing.assert_allclose(A.agg_logit_mean(L, 0.9), softmax(L.mean(0) / 0.9), atol=0, rtol=0)


def test_temperature_applied_before_aggregation():
    L = np.array([[[3.0, 0.0]], [[0.0, 1.0]]])
    T = 0.5
    expected = softmax(L / T).mean(0)
    other = softmax(softmax(L).mean(0) / T)  # aggregate first, temperature after: a different rule
    P = A.agg_mean(L, T)
    np.testing.assert_allclose(P, expected)
    assert not np.allclose(P, other)


def _toy_reg(E=4, n=20, B=30, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(B) * 0.5, size=(E, n))
    dist = BarDistribution(np.linspace(-3, 3, B + 1))
    return probs, dist


@pytest.mark.parametrize("name", ["mixture", "log_pool"])
def test_regression_probs_valid(name):
    probs, dist = _toy_reg()
    pred = A.aggregate_reg(name, probs, dist)
    assert np.isfinite(pred.logp).all()
    np.testing.assert_allclose(np.exp(pred.logp).sum(1), 1.0, atol=1e-9)
    assert (np.diff(pred.quantiles, axis=1) >= -1e-12).all()


@pytest.mark.parametrize("name", ["quantile_mean", "quantile_median"])
def test_regression_quantile_rules_monotone(name):
    probs, dist = _toy_reg(seed=1)
    pred = A.aggregate_reg(name, probs, dist)
    assert (np.diff(pred.quantiles, axis=1) >= -1e-12).all()
    assert np.isfinite(pred.mean).all()


def test_quantile_mean_known_value():
    # bins [0,1],[1,2],[2,3],[3,4]; estimator 1 is uniform on [0,2], estimator 2 uniform on [2,4]
    dist = BarDistribution(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    probs = np.array([[[0.5, 0.5, 0.0, 0.0]], [[0.0, 0.0, 0.5, 0.5]]])
    pred = A.agg_quantile_mean(probs, dist, levels=[0.25, 0.5, 0.75])
    # median of est1 = 1, est2 = 3 -> averaged 2; q25: 0.5 and 2.5 -> 1.5; q75: 1.5, 3.5 -> 2.5
    np.testing.assert_allclose(pred.quantiles, [[1.5, 2.0, 2.5]])
    np.testing.assert_allclose(pred.mean, [2.0])
    mix_same = A.agg_mixture(np.repeat(probs[:1], 2, axis=0), dist, levels=[0.25, 0.5, 0.75])
    one = A.agg_mixture(probs[:1], dist, levels=[0.25, 0.5, 0.75])
    np.testing.assert_allclose(mix_same.quantiles, one.quantiles)
    np.testing.assert_allclose(np.exp(mix_same.logp), np.exp(one.logp))


def test_registry_complete():
    cfg = yaml.safe_load((ROOT / "configs" / "aggregation.yaml").read_text())
    for name in cfg["aggregators"]:
        assert name in A.CLF_AGGREGATORS
    for name in cfg.get("reg_aggregators", []):
        assert name in A.REG_AGGREGATORS
    with pytest.raises(KeyError, match="available"):
        A.get_clf_aggregator("does_not_exist")
    with pytest.raises(KeyError, match="available"):
        A.get_reg_aggregator("does_not_exist")
