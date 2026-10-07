"""Context-row selection (coresets). Every method sees only the arrays it is given — callers pass the training split.

Contract: unique, in-range int64 indices, len == min(budget, n), deterministic given `seed`, and for
classification with budget >= n_classes every class present (except `random`).
Expensive internals are capped by `budget`/`pilot`/`max_clusters`, never by n, so cost stays ~linear in n.
"""

from __future__ import annotations

import warnings

import numpy as np
from threadpoolctl import threadpool_limits

MAX_DIMS = 50  # distance-based methods reduce wider inputs with PCA
MAX_CLUSTERS = 128  # k-means centres per group; larger budgets are spread proportionally over clusters
KMEANS_FIT_ROWS = 4000  # k-means is fit on at most this many rows (assignment is a linear pass)


# --------------------------------------------------------------------------- helpers


def allocate(sizes, budget, min_one=False) -> np.ndarray:
    """Split `budget` across groups proportionally to `sizes` (largest remainder), capped by group size.

    With `min_one`, every non-empty group gets at least one slot when budget >= number of non-empty groups.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    budget = int(min(budget, sizes.sum()))
    out = np.zeros(len(sizes), dtype=np.int64)
    nonempty = sizes > 0
    if min_one and budget >= nonempty.sum():
        out[nonempty] = 1
    remaining = budget - out.sum()
    while remaining > 0:
        room = sizes - out
        if room.sum() == 0:
            break
        share = room / room.sum() * remaining
        add = np.minimum(np.floor(share).astype(np.int64), room)
        left = remaining - add.sum()
        if left > 0:
            frac = np.where(room - add > 0, share - np.floor(share), -1.0)
            for j in np.argsort(-frac, kind="stable")[:left]:
                if room[j] - add[j] > 0:
                    add[j] += 1
        if add.sum() == 0:
            j = int(np.argmax(room))
            add[j] = 1
        out += add
        remaining = budget - out.sum()
    return out


def _groups(y, task, n_bins=10):
    """Row groups: classes for classification, target-quantile bins for regression."""
    y = np.asarray(y)
    if task == "classification":
        labels = np.unique(y)
        return [np.flatnonzero(y == c) for c in labels]
    edges = np.unique(np.quantile(y.astype(float), np.linspace(0, 1, n_bins + 1)[1:-1]))
    codes = np.searchsorted(edges, y.astype(float), side="right")
    return [g for g in (np.flatnonzero(codes == b) for b in range(len(edges) + 1)) if len(g)]


def _sample_groups(groups, alloc, rng):
    parts = [rng.choice(g, int(a), replace=False) for g, a in zip(groups, alloc) if a > 0]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def features(X, seed=0, max_dims=MAX_DIMS) -> np.ndarray:
    """Numeric, NaN-free, standardised features; PCA to at most `max_dims` dimensions (fit on a capped sample)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    X = X.copy()
    nan = np.isnan(X)
    if nan.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X[nan] = np.take(med, np.nonzero(nan)[1])
    X = np.where(np.isfinite(X), X, 0.0)
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    if X.shape[1] > max_dims:
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(seed)
        fit_rows = rng.choice(len(X), min(len(X), 5000), replace=False)
        pca = PCA(n_components=max_dims, svd_solver="randomized", random_state=seed).fit(X[fit_rows])
        X = pca.transform(X)
    return X


def _ensure_class_coverage(idx, y, rng, protected=0):
    """Swap rows so every class appears (budget >= n_classes). Rows idx[:protected] are replaced last."""
    idx = np.array(idx, dtype=np.int64)
    y = np.asarray(y)
    labels = np.unique(y)
    if len(idx) < len(labels):
        return idx
    present = set(np.unique(y[idx]).tolist())
    for c in labels:
        if c in present:
            continue
        counts = {k: v for k, v in zip(*np.unique(y[idx], return_counts=True))}
        # replace a row from the most common class, preferring unprotected positions
        order = list(range(protected, len(idx))) + list(range(protected))
        for pos in order:
            if counts[y[idx[pos]]] > 1:
                chosen = set(idx.tolist())
                cand = np.array([i for i in np.flatnonzero(y == c) if i not in chosen])
                counts[y[idx[pos]]] -= 1
                idx[pos] = rng.choice(cand)
                present.add(c)
                break
    return idx


def _kmeans_select(F, k, rng, seed):
    """Pick k real rows of F: cluster into min(k, MAX_CLUSTERS) centres, allocate k across clusters
    proportionally, take the row nearest each centre first and fill the rest at random within the cluster."""
    from sklearn.cluster import KMeans

    n = len(F)
    if k >= n:
        return np.arange(n)
    n_clusters = int(min(k, MAX_CLUSTERS, n))
    fit_rows = rng.choice(n, min(n, max(KMEANS_FIT_ROWS, 10 * n_clusters)), replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate points -> fewer distinct clusters; empty ones get no rows
        km = KMeans(n_clusters=n_clusters, n_init=1, max_iter=50, random_state=seed).fit(F[fit_rows])
    labels = km.predict(F)
    sizes = np.bincount(labels, minlength=n_clusters)
    alloc = allocate(sizes, k, min_one=True)
    picked = []
    for c in np.flatnonzero(alloc):
        rows = np.flatnonzero(labels == c)
        d = ((F[rows] - km.cluster_centers_[c]) ** 2).sum(axis=1)
        nearest = rows[np.argmin(d)]
        picked.append(nearest)
        if alloc[c] > 1:
            others = rows[rows != nearest]
            picked.extend(rng.choice(others, int(alloc[c]) - 1, replace=False).tolist())
    return np.asarray(picked, dtype=np.int64)


# --------------------------------------------------------------------------- single-subset methods


def sel_random(X, y, budget, rng, task, seed):
    return rng.choice(len(X), budget, replace=False)


def sel_stratified(X, y, budget, rng, task, seed):
    """Class-proportional (classification) or target-quantile-bin proportional (regression, up to 100 bins)."""
    if task == "classification":
        groups = _groups(y, task)
    else:
        groups = _groups(y, task, n_bins=int(min(budget, 100)))
    alloc = allocate([len(g) for g in groups], budget, min_one=task == "classification")
    return _sample_groups(groups, alloc, rng)


def sel_balanced(X, y, budget, rng, task, seed):
    """Equal count per class (per target bin for regression). Shifts the class prior."""
    groups = _groups(y, task)
    sizes = np.array([len(g) for g in groups])
    alloc = np.minimum(sizes, budget // len(groups))
    leftover = budget - alloc.sum()
    order = rng.permutation(len(groups))
    while leftover > 0:
        progressed = False
        for j in order:
            if leftover and alloc[j] < sizes[j]:
                alloc[j] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return _sample_groups(groups, alloc, rng)


def sel_kmeans_stratified(X, y, budget, rng, task, seed):
    """Per-class k-means; return the real row nearest each centroid (regression: per target bin)."""
    F = features(X, seed)
    groups = _groups(y, task)
    alloc = allocate([len(g) for g in groups], budget, min_one=True)
    parts = [g[_kmeans_select(F[g], int(a), rng, seed)] for g, a in zip(groups, alloc) if a > 0]
    return np.concatenate(parts)


def uncertainty_scores(X, y, seed, task, pilot=500, n_trees=25):
    """Uncertainty of every row under a small forest fit on a random pilot subset (entropy / tree variance)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    rng = np.random.default_rng([int(seed), 101])
    n = len(X)
    Xf = np.asarray(X, dtype=np.float64)
    rows = rng.choice(n, min(int(pilot), n), replace=False)
    if task == "classification":
        model = RandomForestClassifier(n_estimators=n_trees, random_state=int(seed), n_jobs=1).fit(Xf[rows], np.asarray(y)[rows])
        P = model.predict_proba(Xf)
        return -(P * np.log(np.clip(P, 1e-12, None))).sum(axis=1)
    model = RandomForestRegressor(n_estimators=n_trees, random_state=int(seed), n_jobs=1).fit(Xf[rows], np.asarray(y)[rows])
    preds = np.stack([t.predict(Xf) for t in model.estimators_], axis=1)
    return preds.var(axis=1)


def sel_uncertainty_mix(X, y, budget, rng, task, seed, frac_hard=0.5, pilot=500):
    """`frac_hard` of the budget from the most uncertain rows under a pilot model, the rest uniformly at random."""
    u = uncertainty_scores(X, y, seed, task, pilot)
    jitter = rng.random(len(u)) * 1e-9 * (np.abs(u).max() + 1.0)
    n_hard = int(round(frac_hard * budget))
    hard = np.argsort(-(u + jitter), kind="stable")[:n_hard]
    rest = np.setdiff1d(np.arange(len(X)), hard)
    soft = rng.choice(rest, budget - n_hard, replace=False)
    idx = np.concatenate([soft, hard])  # random part first so coverage swaps prefer it
    if task == "classification":
        idx = _ensure_class_coverage(idx, y, rng, protected=0)
    return idx


def sel_embedding_kmeans(X, y, budget, rng, task, seed, pilot=1000, backend=None, n_estimators=1):
    """k-means in the backend's embedding space (fit on a random pilot subset); real rows nearest each centre."""
    from .backends import SklearnBackend

    backend = backend or SklearnBackend(n_trees=10)
    n = len(X)
    rows = np.random.default_rng([int(seed), 202]).choice(n, min(int(pilot), n), replace=False)
    if task == "classification":
        rows = np.unique(np.concatenate([rows, [np.flatnonzero(np.asarray(y) == c)[0] for c in np.unique(y)]]))
    E = backend.embed(X[rows], np.asarray(y)[rows], X, seed, task, n_estimators=n_estimators)
    F = features(np.asarray(E).mean(axis=0), seed)
    idx = _kmeans_select(F, budget, rng, seed)
    if task == "classification":
        idx = _ensure_class_coverage(idx, y, rng)
    return idx


CORESET_METHODS = {
    "random": sel_random,
    "stratified": sel_stratified,
    "balanced": sel_balanced,
    "kmeans_stratified": sel_kmeans_stratified,
    "uncertainty_mix": sel_uncertainty_mix,
    "embedding_kmeans": sel_embedding_kmeans,
}
PRIOR_SHIFTING = {"balanced", "uncertainty_mix"}
CLASS_COVERAGE_EXEMPT = {"random"}


def method_info(name) -> dict:
    return {
        "prior_shifting": name in PRIOR_SHIFTING,
        "per_estimator": name in PER_ESTIMATOR_METHODS,
        "class_coverage": name in CORESET_METHODS and name not in CLASS_COVERAGE_EXEMPT,
    }


def _check_budget(budget):
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError(f"budget must be an int, got {type(budget).__name__}")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")


def select_coreset(method, X, y, budget, seed, task="classification", **kw) -> np.ndarray:
    """Indices into the arrays passed in. Pass training arrays only."""
    if method not in CORESET_METHODS:
        raise KeyError(f"unknown coreset method {method!r}; available: {sorted(CORESET_METHODS)}")
    _check_budget(budget)
    n = len(X)
    rng = np.random.default_rng(int(seed))
    if budget >= n:
        return rng.permutation(n).astype(np.int64)
    # one BLAS/OpenMP thread: multithreaded k-means / PCA reductions are not bit-reproducible
    with threadpool_limits(limits=1):
        idx = np.asarray(CORESET_METHODS[method](X, y, int(budget), rng, task, int(seed), **kw), dtype=np.int64)
    if len(idx) != budget or len(np.unique(idx)) != budget:  # pragma: no cover - internal invariant
        raise RuntimeError(f"{method} returned {len(idx)} rows ({len(np.unique(idx))} unique) for budget {budget}")
    return idx


# --------------------------------------------------------------------------- per-estimator methods


def sub_perest_random(n, budget, n_estimators, seed):
    ss = np.random.SeedSequence(int(seed)).spawn(n_estimators)
    return [np.random.default_rng(s).choice(n, min(budget, n), replace=False) for s in ss]


def sub_perest_partition(n, budget, n_estimators, seed):
    """Permute once; estimator e gets a cyclic window. The union covers all rows when E * budget >= n."""
    perm = np.random.default_rng(int(seed)).permutation(n)
    b = min(budget, n)
    step = n / n_estimators if n_estimators * b >= n else b
    return [perm[(int(np.floor(e * step)) + np.arange(b)) % n] for e in range(n_estimators)]


PER_ESTIMATOR_METHODS = {
    "perest_random": sub_perest_random,
    "perest_partition": sub_perest_partition,
}


def per_estimator_subsets(method, X, y, budget, n_estimators, seed, task="classification") -> list[np.ndarray]:
    if method not in PER_ESTIMATOR_METHODS:
        raise KeyError(f"unknown per-estimator method {method!r}; available: {sorted(PER_ESTIMATOR_METHODS)}")
    _check_budget(budget)
    return [np.asarray(s, dtype=np.int64) for s in PER_ESTIMATOR_METHODS[method](len(X), int(budget), int(n_estimators), seed)]


# --------------------------------------------------------------------------- prior correction


def class_prior(y, n_classes) -> np.ndarray:
    counts = np.bincount(np.asarray(y, int), minlength=n_classes).astype(float)
    return counts / counts.sum()


def prior_correct(proba, context_prior, target_prior) -> np.ndarray:
    """Re-weight probabilities by target/context class prior and renormalise (Bayes prior-shift correction).

    tabpfn 9.0.0 ships no `downsample_correction` module, so this is the harness's own implementation.
    Classes absent from the context keep zero mass.
    """
    context_prior = np.asarray(context_prior, float)
    ratio = np.where(context_prior > 0, np.asarray(target_prior, float) / np.maximum(context_prior, 1e-12), 0.0)
    p = np.asarray(proba, float) * ratio
    s = p.sum(axis=1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1.0), proba)
