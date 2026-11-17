"""Context-row selection (coresets). Every method sees only the arrays it is given — callers pass the training split.

Contract: unique, in-range int64 indices, len == min(budget, n), deterministic given `seed`, and for
classification with budget >= n_classes every class present (except `random`).
Expensive internals are capped by `budget`/`pilot`/`max_clusters`, never by n, so cost stays ~linear in n.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np
from threadpoolctl import threadpool_limits

MAX_DIMS = 50  # distance-based methods reduce wider inputs with PCA
MAX_CLUSTERS = 128  # k-means centres per group; larger budgets are spread proportionally over clusters
KMEANS_FIT_ROWS = 4000  # k-means is fit on at most this many rows (assignment is a linear pass)
MIN_CELL_ROWS = 5  # `representative`: at least this many budget rows per stratum on average


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


def _fill(X, med) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    X = X.copy()
    nan = np.isnan(X)
    if nan.any():
        X[nan] = np.take(med, np.nonzero(nan)[1])
    return np.where(np.isfinite(X), X, 0.0)


def feature_transform(X_fit, seed=0, max_dims=MAX_DIMS):
    """The map behind `features`, with NaN medians, scaling and PCA learned from X_fit only, so other rows
    (e.g. test features for routing) land in the same space: `feature_transform(X, s)(X) == features(X, s)`."""
    X = np.asarray(X_fit, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    med = np.zeros(X.shape[1])
    if np.isnan(X).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
    X = _fill(X, med)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    pca = None
    if X.shape[1] > max_dims:
        from sklearn.decomposition import PCA

        Z = (X - mu) / sd
        rng = np.random.default_rng(seed)
        fit_rows = rng.choice(len(Z), min(len(Z), 5000), replace=False)
        pca = PCA(n_components=max_dims, svd_solver="randomized", random_state=seed).fit(Z[fit_rows])

    def apply(Xn) -> np.ndarray:
        Z = (_fill(Xn, med) - mu) / sd
        return Z if pca is None else pca.transform(Z)

    return apply


def features(X, seed=0, max_dims=MAX_DIMS) -> np.ndarray:
    """Numeric, NaN-free, standardised features; PCA to at most `max_dims` dimensions (fit on a capped sample)."""
    return feature_transform(X, seed, max_dims)(X)


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


def _strata(X, y, budget, task, seed, n_cells):
    """Row strata for representative sampling: label group (class / target bin) x k-means cell in feature space.

    Returns (strata, alloc): a list of row-index arrays and the budget share of each, proportional to stratum size
    (classes first, with every class kept when budget >= n_classes, then cells within each class).
    """
    from sklearn.cluster import KMeans

    F = features(X, seed)
    groups = _groups(y, task)
    g_alloc = allocate([len(g) for g in groups], budget, min_one=task == "classification")
    rng = np.random.default_rng([int(seed), 303])
    strata, alloc = [], []
    for g, a in zip(groups, g_alloc):
        if a == 0:
            continue
        k = int(min(n_cells, a // MIN_CELL_ROWS, len(g)))  # few rows per cell -> rounding distorts the shares
        if k <= 1:
            strata.append(g)
            alloc.append(int(a))
            continue
        fit_rows = rng.choice(len(g), min(len(g), max(KMEANS_FIT_ROWS, 10 * k)), replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=int(seed)).fit(F[g][fit_rows])
        labels = km.predict(F[g])
        cells = [g[labels == c] for c in range(k) if (labels == c).any()]
        strata += cells
        alloc += allocate([len(c) for c in cells], int(a)).tolist()
    return strata, np.asarray(alloc, dtype=np.int64)


def sel_representative(X, y, budget, rng, task, seed, n_cells=32):
    """Size the training set down to a representative sample: stratify by class x feature-space cell (k-means)
    and keep rows from every stratum in proportion to its size, at random within each. Keeps the class prior
    and the feature-space density, unlike `kmeans_stratified`, which prefers the row nearest each centre."""
    strata, alloc = _strata(X, y, budget, task, seed, n_cells)
    return _sample_groups(strata, alloc, rng)


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
    "representative": sel_representative,
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


def sub_perest_random(X, y, budget, n_estimators, seed, task):
    n = len(X)
    ss = np.random.SeedSequence(int(seed)).spawn(n_estimators)
    return [np.random.default_rng(s).choice(n, min(budget, n), replace=False) for s in ss]


def sub_perest_partition(X, y, budget, n_estimators, seed, task):
    """Permute once; estimator e gets a cyclic window. The union covers all rows when E * budget >= n."""
    n = len(X)
    perm = np.random.default_rng(int(seed)).permutation(n)
    b = min(budget, n)
    step = n / n_estimators if n_estimators * b >= n else b
    return [perm[(int(np.floor(e * step)) + np.arange(b)) % n] for e in range(n_estimators)]


def sub_perest_representative(X, y, budget, n_estimators, seed, task, n_cells=32):
    """`representative` per estimator: strata are built once, and each estimator draws its own proportional
    sample within them, so every context is representative but the estimators see different rows."""
    n = len(X)
    ss = np.random.SeedSequence(int(seed)).spawn(n_estimators)
    if budget >= n:
        return [np.random.default_rng(s).permutation(n) for s in ss]
    strata, alloc = _strata(X, y, budget, task, seed, n_cells)
    return [_sample_groups(strata, alloc, np.random.default_rng(s)) for s in ss]


def sub_perest_stratified(X, y, budget, n_estimators, seed, task):
    """`stratified` per estimator: class (target-bin) proportional, an independent draw for each estimator."""
    n = len(X)
    ss = np.random.SeedSequence(int(seed)).spawn(n_estimators)
    if budget >= n:
        return [np.random.default_rng(s).permutation(n) for s in ss]
    return [sel_stratified(X, y, budget, np.random.default_rng(s), task, seed) for s in ss]


def _kcenter(F, k, rng) -> np.ndarray:
    """Greedy farthest-point k-center from a random start: every row ends up close to a chosen one."""
    n = len(F)
    picked = [int(rng.integers(n))]
    d = ((F - F[picked[0]]) ** 2).sum(axis=1)
    for _ in range(min(k, n) - 1):
        j = int(np.argmax(d))
        picked.append(j)
        d = np.minimum(d, ((F - F[j]) ** 2).sum(axis=1))
    return np.asarray(picked, dtype=np.int64)


def sub_perest_kcenter(X, y, budget, n_estimators, seed, task):
    """k-center coverage per estimator: within each class (target bin) a farthest-point cover of the feature space
    with that group's proportional share of the budget; estimators differ by their random starting points."""
    n = len(X)
    ss = np.random.SeedSequence(int(seed)).spawn(n_estimators)
    if budget >= n:
        return [np.random.default_rng(s).permutation(n) for s in ss]
    F = features(X, seed)
    groups = _groups(y, task)
    alloc = allocate([len(g) for g in groups], budget, min_one=task == "classification")
    out = []
    for s in ss:
        rng = np.random.default_rng(s)
        out.append(np.concatenate([g[_kcenter(F[g], int(a), rng)] for g, a in zip(groups, alloc) if a > 0]))
    return out


def _expert_clusters(X, n_clusters, seed):
    """k-means of the training rows into one cluster per estimator: (feature map, model, train labels, train features)."""
    from sklearn.cluster import KMeans

    T = feature_transform(X, seed)
    F = T(X)
    rng = np.random.default_rng([int(seed), 404])
    fit_rows = rng.choice(len(F), min(len(F), KMEANS_FIT_ROWS), replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=int(min(n_clusters, len(fit_rows))), n_init=3, random_state=int(seed)).fit(F[fit_rows])
    return T, km, km.predict(F), F


def sub_cluster_experts(X, y, budget, n_estimators, seed, task):
    """Estimator e specialises on training cluster e: `budget` random rows of the cluster, topped up with the
    rows of other clusters nearest its centre when the cluster is smaller than the budget."""
    n = len(X)
    _, km, labels, F = _expert_clusters(X, n_estimators, seed)
    out = []
    for e, s in enumerate(np.random.SeedSequence(int(seed)).spawn(n_estimators)):
        rng = np.random.default_rng(s)
        c = e % len(km.cluster_centers_)
        own = np.flatnonzero(labels == c)
        if len(own) >= budget:
            idx = rng.choice(own, budget, replace=False)
        else:
            d = ((F - km.cluster_centers_[c]) ** 2).sum(axis=1)
            d[own] = np.inf
            idx = np.concatenate([own, np.argsort(d, kind="stable")[: min(budget, n) - len(own)]])
        if task == "classification":
            idx = _ensure_class_coverage(idx, y, rng)
        out.append(idx)
    return out


def gate_weights(d, tau, bias=None) -> np.ndarray:
    """(E, n) estimator weights per row: softmax over estimators of -d / tau (+ a per-estimator bias)."""
    z = -(d - d.min(axis=0, keepdims=True)) / tau
    if bias is not None:
        z = z + np.asarray(bias, np.float64)[:, None]
    z = z - z.max(axis=0, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=0, keepdims=True)


def cluster_gate_distances(X_train, X_evals, n_estimators, seed) -> tuple[list[np.ndarray], float]:
    """Squared distances (E, n) of each evaluation set's rows to the `cluster_experts` centres of X_train, and the
    heuristic gate temperature: the median squared distance of training rows to their own centre."""
    with threadpool_limits(limits=1):
        T, km, labels, F = _expert_clusters(X_train, n_estimators, seed)
        C = km.cluster_centers_
        tau = float(np.median(((F - C[labels]) ** 2).sum(axis=1))) or 1.0
        order = np.arange(n_estimators) % len(C)
        ds = [((T(Xe)[None, :, :] - C[:, None, :]) ** 2).sum(axis=2)[order] for Xe in X_evals]
    return ds, tau


def cluster_gate(X_train, X_test, n_estimators, seed) -> np.ndarray:
    """(E, n_test) weights for `cluster_experts`: softmax over estimators of -d(x, centre_e)^2 / tau, with tau the
    median squared distance of training rows to their own centre. Uses test features only."""
    (d,), tau = cluster_gate_distances(X_train, [X_test], n_estimators, seed)
    return gate_weights(d, tau)


GATE_TAU_GRID = np.logspace(-2, 2, 41)  # multipliers of the heuristic temperature tried by `fit_gate`
GATE_TAU_GRID_WIDE = np.logspace(-4, 2, 61)  # same steps, reaching down to nearest-expert routing


def _gate_objective(w, score, y=None):
    """(loss, d loss / d w) of gate weights w (E, n) on held-out rows. With y None, `score` is each estimator's
    log-likelihood of the true label and the loss is the mean negative log-likelihood of the mixture; otherwise
    `score` is each estimator's predicted mean and the loss is the squared error of the mixture mean over var(y)."""
    n = w.shape[1]
    if y is None:
        shift = score.max(axis=0)
        p = np.exp(score - shift)
        mix = (w * p).sum(axis=0)
        return float(-np.mean(np.log(mix) + shift)), -p / mix / n
    scale = float(np.var(y)) or 1.0
    r = (w * score).sum(axis=0) - y
    return float(np.mean(r**2) / scale), 2 * r * score / (n * scale)


def gate_loss(d, score, tau, bias=None, y=None) -> float:
    return _gate_objective(gate_weights(d, tau, bias), score, y)[0]


def fit_gate(d, score, tau0, y=None, bias=False, ridge=1e-3, grid=None) -> dict:
    """Fit the gate on held-out *training* rows. `d` (E, n): squared distances to the expert centres. `score` (E, n):
    each expert's log-likelihood of the row's training label (classification), or with `y` its predicted mean, the
    gate then minimising the squared error of the mixture mean (regression); see `_gate_objective`.

    The temperature is a grid search over `grid * tau0`, `grid` defaulting to `GATE_TAU_GRID` (ties go to the
    multiplier nearest 1). With `bias`,
    a per-expert offset is then fitted jointly with the temperature (L-BFGS, ridge penalty on the offsets), starting
    from the grid solution. Returns {"tau", "bias", "loss", "loss_heuristic", "nearest_is_best"}.
    """
    d = np.asarray(d, np.float64)
    score = np.asarray(score, np.float64)
    if y is None:
        score = np.maximum(score, -50.0)
        best_expert = score.argmax(axis=0)
    else:
        y = np.asarray(y, np.float64)
        best_expert = np.abs(score - y).argmin(axis=0)
    E = d.shape[0]
    grid = GATE_TAU_GRID if grid is None else np.asarray(grid, np.float64)
    losses = np.array([gate_loss(d, score, m * tau0, y=y) for m in grid])
    best = np.flatnonzero(losses <= losses.min() + 1e-12)
    k = best[np.argmin(np.abs(np.log(grid[best])))]
    res = {
        "tau": float(grid[k] * tau0),
        "bias": None,
        "loss": float(losses[k]),
        "loss_heuristic": gate_loss(d, score, tau0, y=y),
        "nearest_is_best": float(np.mean(d.argmin(axis=0) == best_expert)),
    }
    if not bias:
        return res
    from scipy.optimize import minimize

    dd = d - d.min(axis=0, keepdims=True)

    def objective(theta):
        s, b = theta[0], theta[1:]
        w = gate_weights(d, float(np.exp(s)), b)
        loss, gw = _gate_objective(w, score, y)
        g = w * (gw - (w * gw).sum(axis=0, keepdims=True))  # d loss / d z, with w = softmax(z), z = -dd / tau + b
        grad = np.concatenate([[(g * dd).sum() * np.exp(-s)], g.sum(axis=1) + 2 * ridge * b])
        return loss + ridge * float(b @ b), grad

    lo, hi = np.log(grid[0] * tau0), np.log(grid[-1] * tau0)
    opt = minimize(objective, np.concatenate([[np.log(res["tau"])], np.zeros(E)]), jac=True, method="L-BFGS-B", bounds=[(lo, hi)] + [(-10, 10)] * E)
    tau, b = float(np.exp(opt.x[0])), opt.x[1:]
    loss = gate_loss(d, score, tau, b, y=y)
    if loss < res["loss"]:
        res.update(tau=tau, bias=b, loss=loss)
    else:  # never worse than the temperature-only gate on the rows it was fitted on
        res["bias"] = np.zeros(E)
    return res


def gate_holdout(y, seed, task, val_fraction=0.2, val_max=2000) -> tuple[np.ndarray, np.ndarray]:
    """(context_pool_idx, val_idx): a seeded random slice of the training rows kept out of every expert's context so
    the gate can be fitted on it. Every class keeps at least one row in the context pool."""
    n = len(y)
    rng = np.random.default_rng([int(seed), 606])
    n_val = int(min(val_max, round(val_fraction * n)))
    val = np.zeros(n, bool)
    val[rng.permutation(n)[:n_val]] = True
    if task == "classification":
        y = np.asarray(y)
        for c in np.unique(y):
            rows = np.flatnonzero(y == c)
            if val[rows].all():
                val[rows[0]] = False
    return np.flatnonzero(~val), np.flatnonzero(val)


PER_ESTIMATOR_METHODS = {
    "perest_random": sub_perest_random,
    "perest_partition": sub_perest_partition,
    "perest_representative": sub_perest_representative,
    "perest_stratified": sub_perest_stratified,
    "perest_kcenter": sub_perest_kcenter,
    "cluster_experts": sub_cluster_experts,
}


# --------------------------------------------------------------------------- local experts (routed by own features)


def _sq_dists(F, C) -> np.ndarray:
    """(len(C), len(F)) squared Euclidean distances, without a (K, n, dim) intermediate."""
    F = np.asarray(F, np.float64)
    C = np.asarray(C, np.float64)
    d = (C**2).sum(axis=1)[:, None] + (F**2).sum(axis=1)[None, :] - 2.0 * (C @ F.T)
    return np.maximum(d, 0.0)


def supervised_feature_scale(F, y, task, seed, max_rows=5000, n_trees=50, floor=0.1) -> np.ndarray:
    """Per-feature scale that turns squared distance into an importance-weighted one: sqrt(dim * importance).

    Importances come from a random forest fitted on a capped sample of the rows and labels it is given (callers pass
    training rows), mixed with a share `floor` of uniform so no feature vanishes. The squared scales average 1, so
    distances stay on the scale of the unweighted space."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    F = np.asarray(F, np.float64)
    y = np.asarray(y)
    dim = F.shape[1]
    rows = np.random.default_rng([int(seed), 707]).choice(len(F), min(len(F), int(max_rows)), replace=False)
    forest_cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    forest = forest_cls(n_estimators=int(n_trees), min_samples_leaf=5, max_features="sqrt", random_state=int(seed), n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = np.asarray(forest.fit(F[rows], y[rows]).feature_importances_, np.float64)
    if not np.isfinite(imp).all() or imp.sum() <= 0:
        imp = np.full(dim, 1.0 / dim)
    imp = (1.0 - floor) * imp / imp.sum() + floor / dim
    return np.sqrt(dim * imp)


@dataclass
class ExpertPlan:
    """Local experts built from training rows: k-means centres, one context per centre, and for every centre the
    centres nearest to it (itself first), i.e. the experts that predict the rows whose nearest centre it is."""

    space: object  # callable: raw rows -> the feature space the centres live in
    centres: np.ndarray  # (K, dim)
    contexts: list  # K index arrays into the rows the plan was built from
    neighbours: np.ndarray  # (K, n_estimators) centre ids
    tau0: float  # heuristic gate temperature: median squared distance of a row to its own centre
    scale: np.ndarray | None  # per-feature scale applied before clustering (None: unweighted)


def expert_plan(X, y, budget, n_estimators, seed, task, weighted=False, coverage=1.0, max_clusters=128) -> ExpertPlan:
    """Cluster the rows it is given (callers pass training rows) into about coverage * n / budget clusters, at least
    `n_estimators` and at most `max_clusters`, and build one `budget`-row context per cluster: `budget` random rows
    of the cluster, or all of them topped up with the nearest rows of other clusters (as in `cluster_experts`).

    `coverage` 1 gives clusters of about `budget` rows. With 2 they hold about half that, so every context reaches
    into the neighbouring clusters and a row near a cluster border has its neighbours in one context.
    `weighted`: cluster in the space scaled by `supervised_feature_scale` (uses y), so features that say nothing
    about the target stop deciding what is near."""
    _check_budget(budget)
    from sklearn.cluster import KMeans

    n = len(X)
    b = min(int(budget), n)
    with threadpool_limits(limits=1):
        T = feature_transform(X, seed)
        F = T(X)
        scale = supervised_feature_scale(F, y, task, seed) if weighted else None
        if scale is not None:
            F = F * scale
        K = max(int(n_estimators), min(int(max_clusters), int(np.ceil(coverage * n / b))))
        fit_rows = np.random.default_rng([int(seed), 404]).choice(n, min(n, max(KMEANS_FIT_ROWS, 50 * K)), replace=False)
        if K > len(fit_rows):
            raise ValueError(f"local experts need at least n_estimators={n_estimators} clusters, but there are only {n} rows")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C = KMeans(n_clusters=K, n_init=3, random_state=int(seed)).fit(F[fit_rows]).cluster_centers_
        d = _sq_dists(F, C)  # (K, n)
        labels = d.argmin(axis=0)
        tau0 = float(np.median(d[labels, np.arange(n)])) or 1.0
        contexts = []
        for c, s in enumerate(np.random.SeedSequence([int(seed), 808]).spawn(K)):
            rng = np.random.default_rng(s)
            own = np.flatnonzero(labels == c)
            if len(own) >= b:
                idx = rng.choice(own, b, replace=False)
            else:
                order = np.argsort(d[c], kind="stable")
                idx = np.concatenate([own, order[labels[order] != c][: b - len(own)]])
            if task == "classification":
                idx = _ensure_class_coverage(idx, y, rng, protected=min(len(own), b))
            contexts.append(np.asarray(idx, dtype=np.int64))
        dc = _sq_dists(C, C)
        np.fill_diagonal(dc, -1.0)  # a centre is its own nearest expert
        neighbours = np.argsort(dc, axis=1, kind="stable")[:, : int(n_estimators)]
    space = T if scale is None else (lambda Xn: T(Xn) * scale)
    return ExpertPlan(space, C, contexts, neighbours, tau0, scale)


def expert_route(plan: ExpertPlan, X_eval) -> tuple[np.ndarray, np.ndarray]:
    """(home (n,), d (n_estimators, n)): each row's nearest centre and its squared distances to that centre's
    experts, `plan.neighbours[home]`, the home centre first. A row is routed by its own features only."""
    with threadpool_limits(limits=1):
        D = _sq_dists(plan.space(X_eval), plan.centres)
    home = D.argmin(axis=0)
    return home, D[plan.neighbours[home].T, np.arange(D.shape[1])]


# --------------------------------------------------------------------------- local contexts (routed by test features)


def _test_groups(X_train, X_test, seed, n_groups):
    """k-means of the test rows (features only) in the train-fitted feature space: (F_train, F_test, labels, centres)."""
    from sklearn.cluster import KMeans

    T = feature_transform(X_train, seed)
    Ftr, Fte = T(X_train), T(X_test)
    m = len(Fte)
    G = int(max(1, min(n_groups, m)))
    fit_rows = np.random.default_rng([int(seed), 505]).choice(m, min(m, KMEANS_FIT_ROWS), replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=min(G, len(fit_rows)), n_init=3, random_state=int(seed)).fit(Fte[fit_rows])
    return Ftr, Fte, km.predict(Fte), km.cluster_centers_


def local_mix_routing(X_train, y_train, X_test, budget, n_estimators, seed, task="classification", ratio=0.9, n_groups=8, pool_factor=2):
    """Group the test rows by their features (k-means, never labels) and build, per group and estimator, a context of
    round(ratio * budget) rows drawn from the pool_factor-times-larger set of training rows nearest the group centre
    plus random rows from the rest of the training split. Returns [(test_row_indices, [subset per estimator])].

    The contexts depend on the whole test batch, so results built on this are transductive.
    """
    if not 0 <= ratio <= 1:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    _check_budget(budget)
    n = len(X_train)
    with threadpool_limits(limits=1):
        Ftr, _, labels, centres = _test_groups(X_train, X_test, seed, n_groups)
        b = min(int(budget), n)
        n_local = round(ratio * b)
        out = []
        for g in range(len(centres)):
            rows = np.flatnonzero(labels == g)
            if not len(rows):
                continue
            order = np.argsort(((Ftr - centres[g]) ** 2).sum(axis=1), kind="stable")
            pool = order[: min(n, max(n_local, pool_factor * n_local))]
            subsets = []
            for s in np.random.SeedSequence([int(seed), g]).spawn(n_estimators):
                rng = np.random.default_rng(s)
                if b >= n:
                    subsets.append(rng.permutation(n))
                    continue
                local = rng.choice(pool, n_local, replace=False)
                rest = rng.choice(np.setdiff1d(np.arange(n), local), b - n_local, replace=False)
                idx = np.concatenate([rest, local]).astype(np.int64)  # random part first: coverage swaps use it first
                if task == "classification":
                    idx = _ensure_class_coverage(idx, y_train, rng)
                subsets.append(idx)
            out.append((rows, subsets))
    return out


def _rbf_bandwidth(F, seed, n_rows=1000) -> float:
    """Median pairwise distance of a seeded sample of rows (the median heuristic)."""
    rows = np.random.default_rng([int(seed), 606]).choice(len(F), min(len(F), n_rows), replace=False)
    Z = F[rows]
    d = np.sqrt(np.clip((Z**2).sum(1)[:, None] + (Z**2).sum(1)[None, :] - 2 * Z @ Z.T, 0, None))
    d = d[np.triu_indices(len(Z), 1)]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def kernel_mean(F_cand, F_target, gamma, chunk=256) -> np.ndarray:
    """mu(x) = mean over target rows q of exp(-gamma |x - q|^2), for every candidate row x."""
    a2 = (F_cand**2).sum(axis=1)
    mu = np.zeros(len(F_cand))
    for i in range(0, len(F_target), chunk):
        Q = F_target[i : i + chunk]
        D = np.clip(a2[:, None] + (Q**2).sum(axis=1)[None, :] - 2 * F_cand @ Q.T, 0, None)
        mu += np.exp(-gamma * D).sum(axis=1)
    return mu / len(F_target)


def kernel_herding(F_cand, mu, k, gamma) -> np.ndarray:
    """Greedy MMD minimisation: repeatedly add the candidate maximising mu(x) - mean over picked s of k(x, s), which
    is the exact greedy step for an RBF kernel (k(x, x) = 1). Deterministic. Indices into F_cand, in picking order."""
    n = len(F_cand)
    a2 = (F_cand**2).sum(axis=1)
    acc = np.zeros(n)
    picked = np.zeros(n, dtype=bool)
    out = []
    for t in range(min(int(k), n)):
        score = mu - acc / (t + 1)
        score[picked] = -np.inf
        j = int(np.argmax(score))
        out.append(j)
        picked[j] = True
        acc += np.exp(-gamma * np.maximum(a2 + a2[j] - 2.0 * (F_cand @ F_cand[j]), 0.0))
    return np.asarray(out, dtype=np.int64)


def crumb_routing(X_train, y_train, X_test, budget, n_estimators, seed, task="classification", n_groups=8, per_estimator=False, keep_frac=0.7):
    """CRUMB (Heredge et al., arXiv 2606.11473): k-means the test rows (features only), then per group choose the
    training rows whose distribution matches the group by kernel herding (RBF, median-heuristic bandwidth on the
    train-fitted feature space). `per_estimator=False`: one herded context shared by all estimators, as in the paper.
    `per_estimator=True`: estimator e herds from its own random `keep_frac` of the training rows. Kernel, bandwidth
    and class coverage are this harness's choices. Returns [(test_row_indices, [subset per estimator])]; transductive.
    """
    _check_budget(budget)
    n = len(X_train)
    with threadpool_limits(limits=1):
        Ftr, Fte, labels, centres = _test_groups(X_train, X_test, seed, n_groups)
        gamma = 1.0 / (2.0 * _rbf_bandwidth(Ftr, seed) ** 2)
        b = min(int(budget), n)
        out = []
        for g in range(len(centres)):
            rows = np.flatnonzero(labels == g)
            if not len(rows):
                continue
            ss = np.random.SeedSequence([int(seed), g, 606]).spawn(n_estimators)
            if b >= n:
                out.append((rows, [np.random.default_rng(s).permutation(n) for s in ss]))
                continue
            mu = kernel_mean(Ftr, Fte[rows], gamma)

            def herd(cand, rng):
                idx = cand[kernel_herding(Ftr[cand], mu[cand], b, gamma)]
                if task == "classification":  # swap the last-herded rows first
                    idx = _ensure_class_coverage(idx[::-1], y_train, rng)[::-1].copy()
                return idx

            if per_estimator:
                keep = max(b, round(keep_frac * n))
                subsets = []
                for s in ss:
                    rng = np.random.default_rng(s)
                    subsets.append(herd(np.sort(rng.choice(n, keep, replace=False)), rng))
            else:
                idx = herd(np.arange(n), np.random.default_rng(ss[0]))
                subsets = [idx.copy() for _ in range(n_estimators)]
            out.append((rows, subsets))
    return out


ROUTED_METHODS = {
    "crumb": partial(crumb_routing, per_estimator=False),
    "crumb_perest": partial(crumb_routing, per_estimator=True),
}


def per_estimator_subsets(method, X, y, budget, n_estimators, seed, task="classification", **kw) -> list[np.ndarray]:
    if method not in PER_ESTIMATOR_METHODS:
        raise KeyError(f"unknown per-estimator method {method!r}; available: {sorted(PER_ESTIMATOR_METHODS)}")
    _check_budget(budget)
    with threadpool_limits(limits=1):
        subsets = PER_ESTIMATOR_METHODS[method](X, y, int(budget), int(n_estimators), int(seed), task, **kw)
    return [np.asarray(s, dtype=np.int64) for s in subsets]


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
