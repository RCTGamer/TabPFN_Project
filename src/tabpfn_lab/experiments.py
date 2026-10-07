"""Experiment runners: A (aggregation) and B (coresets), with caching, resume, sharding and failure logging.

A run is a stable, ordered list of cells. Each cell is computed, its rows appended to the CSV, and skipped on
rerun. Exceptions inside a cell are recorded in `<name>.failures.csv` and the sweep continues.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregation as agg
from . import coresets as cs
from .backends import (
    Backend,
    ClfOutputs,
    RegOutputs,
    gpu_peak_mb,
    is_oom_error,
    make_backend,
    reset_gpu_peak,
    translate_probs,
)
from .cache import OutputCache
from .datasets import Split, load_split
from .metrics import CLF_METRICS, REG_METRICS, score_classification, score_regression
from .sizes import budget_keys, learning_curve_budgets, tier_of
from .validate import COLUMNS

log = logging.getLogger("tabpfn_lab")

BUILTIN_METHODS = {"builtin_subsample": "auto", "builtin_majority_downsample": "majority_downsample"}
CANARY = "canary_shuffled"
REFERENCE_METHODS_B = {"full", "random", CANARY, *BUILTIN_METHODS}
REQUIRED_BASELINES_B = ["full", "random", "builtin_subsample", "builtin_majority_downsample"]
DEFAULT_REG_AGGREGATORS = ["mixture", "log_pool", "quantile_mean", "quantile_median"]


class TuningOnConfirmError(ValueError):
    """Raised when tuning code is handed a confirm dataset."""


# --------------------------------------------------------------------------- config


def validate_config(cfg: dict) -> None:
    ds = cfg.get("datasets")
    if isinstance(ds, dict):
        dev = {d["name"] for d in ds.get("dev", []) or []}
        confirm = {d["name"] for d in ds.get("confirm", []) or []}
        overlap = dev & confirm
        if overlap:
            raise ValueError(f"dev and confirm datasets overlap: {sorted(overlap)}")


def dataset_specs(cfg: dict) -> list[dict]:
    """Flatten the dataset list; each spec gets a `split` of 'dev' or 'confirm'."""
    ds = cfg["datasets"]
    out = []
    if isinstance(ds, dict):
        for split in ("dev", "confirm"):
            for d in ds.get(split, []) or []:
                out.append({**d, "split": split})
    else:
        for d in ds:
            out.append({**d, "split": d.get("split", "dev")})
    names = [d["name"] for d in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate dataset names: {names}")
    return out


def _seed_list(v, default):
    if v is None:
        return list(default)
    if isinstance(v, int):
        return list(range(v))
    return [int(s) for s in v]


def normalize_config(cfg: dict, experiment: str) -> dict:
    validate_config(cfg)
    c = dict(cfg)
    c.setdefault("name", experiment)
    c.setdefault("backend", {"name": "sklearn"})
    c.setdefault("test_size", 0.3)
    c.setdefault("n_estimators", 8)
    c.setdefault("aggregators", ["mean"])
    c.setdefault("reg_aggregators", DEFAULT_REG_AGGREGATORS)
    c.setdefault("aggregator_kwargs", {})
    c.setdefault("references", ["native", "native_tuned"])
    c.setdefault("cache_dir", "results/raw")
    c.setdefault("log_dir", "results/logs")
    c.setdefault("quantile_levels", list(agg.DEFAULT_LEVELS))
    c.setdefault("val_fraction", 0.2)
    c["seeds"] = _seed_list(c.get("seeds"), [0])
    if experiment == "coreset":
        c["selection_seeds"] = _seed_list(c.get("selection_seeds"), c["seeds"])
        c["split_seeds"] = _seed_list(c.get("split_seeds"), [0])
        c.setdefault("budget_fractions", [0.1])
        c.setdefault("budget_absolute", [])
        c.setdefault("coreset_methods", ["random"])
        c.setdefault("coreset_kwargs", {})
        c.setdefault("aggregator", "mean")
        c.setdefault("learning_curve_points", 6)
        c.setdefault("combined_aggregators", [])
    c.setdefault("out_csv", f"results/tables/{c['name']}_{experiment}.csv")
    c["experiment"] = experiment
    return c


def _n_estimators_list(cfg) -> list[int]:
    v = cfg["n_estimators"]
    return sorted({int(x) for x in v}) if isinstance(v, (list, tuple)) else [int(v)]


def shard_path(out_csv: str, shard: tuple[int, int] | None) -> Path:
    p = Path(out_csv)
    return p if shard is None else p.with_name(f"{p.stem}.shard{shard[0]}{p.suffix}")


def failures_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}.failures.csv")


def meta_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}.meta.json")


def read_results(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"flags": str, "budget_key": str, "cell": str, "dataset": str, "method": str})
    for c in ("flags", "budget_key"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def setup_logging(name: str, log_dir) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tabpfn_lab")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        if getattr(h, "_tabpfn_lab", False):
            logger.removeHandler(h)
            h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")  # flushed on every record
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        h._tabpfn_lab = True
        logger.addHandler(h)
    return logger


# --------------------------------------------------------------------------- cells and context


@dataclass(frozen=True)
class Cell:
    experiment: str
    dataset: str
    split_seed: int
    seed: int
    method: str = ""
    budget: int | None = None
    budget_key: str = ""

    @property
    def id(self) -> str:
        return f"{self.experiment}|{self.dataset}|s{self.split_seed}|r{self.seed}|{self.method}|{self.budget_key}|{self.budget}"


class Context:
    """Per-run state: backend, cache, splits, and run metadata."""

    def __init__(self, cfg: dict, backend: Backend | None):
        self.cfg = cfg
        self.backend = backend if backend is not None else make_backend(cfg["backend"])
        self.cache = OutputCache(cfg["cache_dir"]) if cfg.get("cache_dir") else None
        self.specs = {d["name"]: d for d in dataset_specs(cfg)}
        self._splits: dict = {}
        self.backend_meta: dict = {}
        self.split_hashes: dict = {}

    def split(self, name: str, split_seed: int) -> Split:
        key = (name, split_seed)
        if key not in self._splits:
            s = load_split(self.specs[name], split_seed, self.cfg["test_size"])
            self._splits[key] = s
            self.split_hashes[f"{name}|{split_seed}"] = s.test_hash
        return self._splits[key]

    def _key(self, split: Split, n_estimators, seed, subset_id, **extra):
        return OutputCache.make_key(
            split.name, split.task, split.split_seed, split.n_train, split.n_test, self.backend.params(), n_estimators, subset_id, seed=seed, **extra
        )

    def _cached(self, key, compute, to_arrays, from_arrays):
        if self.cache is not None and self.cache.exists(key):
            arrays, meta = self.cache.get(key)
            return from_arrays(arrays, meta)
        obj, meta = compute()
        if self.cache is not None:
            self.cache.put(key, to_arrays(obj), meta)
        return obj

    def outputs(self, split: Split, X, y, n_estimators, seed, subset_id, overrides=None, X_eval=None):
        """Per-estimator outputs of a backend fit on (X, y), evaluated on the split's test rows (or X_eval)."""
        X_eval = split.X_test if X_eval is None else X_eval
        key = self._key(split, n_estimators, seed, subset_id, overrides=overrides or {})

        def compute():
            if split.task == "classification":
                o = self.backend.clf_outputs(X, y, X_eval, split.classes, n_estimators, seed, overrides=overrides)
            else:
                o = self.backend.reg_outputs(X, y, X_eval, n_estimators, seed, overrides=overrides)
            if not self.backend_meta:
                self.backend_meta = dict(o.meta)
            return o, o.meta

        cls = ClfOutputs if split.task == "classification" else RegOutputs
        return self._cached(key, compute, lambda o: o.to_arrays(), lambda a, m: cls.from_arrays(a, m))

    def native(self, split: Split, n_estimators, seed, tuned=False):
        key = self._key(split, n_estimators, seed, f"native{'_tuned' if tuned else ''}")
        levels = self.cfg["quantile_levels"]

        def compute():
            if split.task == "classification":
                return {"proba": self.backend.native_clf(split.X_train, split.y_train, split.X_test, split.classes, n_estimators, seed, tuned=tuned)}, {}
            return self.backend.native_reg(split.X_train, split.y_train, split.X_test, n_estimators, seed, levels), {}

        return self._cached(key, compute, lambda d: d, lambda a, m: a)

    def validation_slice(self, split: Split, seed):
        """Train-only validation slice: (train_sub_idx, val_idx) into the training split."""
        from sklearn.model_selection import train_test_split

        y = split.y_train
        strat = y if split.task == "classification" and np.bincount(y).min() >= 2 else None
        return train_test_split(np.arange(split.n_train), test_size=self.cfg["val_fraction"], random_state=seed, stratify=strat)

    def val_outputs(self, split: Split, n_estimators, seed):
        tr, va = self.validation_slice(split, seed)
        out = self.outputs(
            split, split.X_train[tr], split.y_train[tr], n_estimators, seed, f"val:{self.cfg['val_fraction']}", X_eval=split.X_train[va]
        )
        return out, split.y_train[va]


def _rows(cfg, cell: Cell, split: Split, spec, backend_name, method, metrics: dict, role, budget=None, budget_key="", n_estimators=None):
    n_train = split.n_train
    return [
        {
            "experiment": cfg["experiment"],
            "dataset": split.name,
            "split": spec.get("split", "dev"),
            "tier": tier_of(n_train),
            "task": split.task,
            "seed": int(cell.seed),
            "method": method,
            "budget": float("nan") if budget is None else float(budget),
            "ratio": float("nan") if budget is None else float(budget) / n_train,
            "n_train": int(n_train),
            "metric": metric,
            "value": float(value),
            "backend": backend_name,
            "role": role,
            "flags": "",
            "split_seed": int(cell.split_seed),
            "n_estimators": int(n_estimators if n_estimators is not None else -1),
            "budget_key": budget_key,
            "cell": cell.id,
        }
        for metric, value in metrics.items()
    ]


def _score(split: Split, pred, levels) -> dict:
    if split.task == "classification":
        return score_classification(split.y_test, pred)
    return score_regression(split.y_test, pred.mean, pred.quantiles, levels)


# --------------------------------------------------------------------------- generic runner


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _done_cells(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["cell"], dtype={"cell": str})["cell"])


def _run_cells(cfg, ctx: Context, cells: list[Cell], run_cell, shard, meta: dict) -> Path:
    csv = shard_path(cfg["out_csv"], shard)
    fail = failures_path(csv)
    name = cfg["name"] + (f".shard{shard[0]}" if shard else "")
    setup_logging(name, cfg["log_dir"])
    if shard is not None:
        i, n = shard
        if not 0 <= i < n:
            raise ValueError(f"bad shard {i}/{n}")
        cells = [c for j, c in enumerate(cells) if j % n == i]
    done = _done_cells(csv)
    meta = {**meta, "shard": list(shard) if shard else None, "n_cells": len(cells)}
    _write_meta(csv, meta)
    log.info("run %s: %d cells (%d already done) -> %s", name, len(cells), len(done & {c.id for c in cells}), csv)
    for k, cell in enumerate(cells):
        if cell.id in done:
            continue
        t0 = time.time()
        reset_gpu_peak()
        try:
            rows = run_cell(cell)
        except Exception as e:  # failures are data: record and continue
            err = {
                "cell": cell.id,
                "dataset": cell.dataset,
                "split_seed": cell.split_seed,
                "seed": cell.seed,
                "method": cell.method,
                "budget": cell.budget,
                "error_type": type(e).__name__,
                "oom": is_oom_error(e),
                "message": str(e)[:2000],
            }
            fail.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([err]).to_csv(fail, mode="a", header=not fail.exists(), index=False)
            log.error("cell %d/%d FAILED %s: %s: %s\n%s", k + 1, len(cells), cell.id, type(e).__name__, e, traceback.format_exc(limit=3))
            continue
        _append(csv, rows)
        log.info("cell %d/%d %s rows=%d wall=%.2fs peak_gpu_mb=%.0f", k + 1, len(cells), cell.id, len(rows), time.time() - t0, gpu_peak_mb())
    meta["backend_meta"] = ctx.backend_meta
    meta["split_hashes"] = ctx.split_hashes
    _write_meta(csv, meta)
    return csv


def _write_meta(csv: Path, meta: dict) -> None:
    from .backends import _jsonable

    p = meta_path(csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(meta), indent=2, sort_keys=True))
    tmp.replace(p)


def _tabpfn_version_meta(backend: Backend) -> str | None:
    if getattr(backend, "name", "").startswith("tabpfn"):
        import importlib.metadata as md

        try:
            return md.version("tabpfn")
        except md.PackageNotFoundError:
            return None
    return None


# --------------------------------------------------------------------------- experiment A


def aggregation_methods(cfg) -> dict[str, list[str]]:
    refs = cfg["references"]
    return {
        "classification": ["single", *cfg["aggregators"], *refs],
        "regression": ["single", *cfg["reg_aggregators"], *[r for r in refs if r == "native"]],
    }


def aggregation_cells(cfg, ctx) -> list[Cell]:
    return [Cell("aggregation", name, s, s) for name in ctx.specs for s in cfg["seeds"]]


def _aggregation_cell(cfg, ctx: Context, cell: Cell) -> list[dict]:
    split = ctx.split(cell.dataset, cell.split_seed)
    spec = ctx.specs[cell.dataset]
    e_list = _n_estimators_list(cfg)
    e_max = max(e_list)
    levels = cfg["quantile_levels"]
    bname = ctx.backend.name
    rows: list[dict] = []
    out = ctx.outputs(split, split.X_train, split.y_train, e_max, cell.seed, "full")
    if split.task == "classification":
        for a in cfg["aggregators"]:
            agg.get_clf_aggregator(a)
        val = None
        if any(a in agg.NEEDS_VALIDATION for a in cfg["aggregators"]):
            val = ctx.val_outputs(split, e_max, cell.seed)
        for E in e_list:
            L, T = out.logits[:E], out.temperature
            rows += _rows(cfg, cell, split, spec, bname, "single", _score(split, agg.aggregate_clf("mean", L[:1], T), levels), "reference", n_estimators=E)
            for a in cfg["aggregators"]:
                kw = dict(cfg["aggregator_kwargs"].get(a, {}) or {})
                if a in agg.NEEDS_VALIDATION:
                    kw.update(val_logits=val[0].logits[:E], val_y=val[1])
                if a in agg.NEEDS_WEIGHTS and "w" in kw:
                    kw["w"] = np.asarray(kw["w"], float)[:E]
                P = agg.aggregate_clf(a, L, T, **kw)
                rows += _rows(cfg, cell, split, spec, bname, a, _score(split, P, levels), "baseline" if a == "mean" else "method", n_estimators=E)
            for ref in cfg["references"]:
                P = ctx.native(split, E, cell.seed, tuned=(ref == "native_tuned"))["proba"]
                rows += _rows(cfg, cell, split, spec, bname, ref, _score(split, P, levels), "reference", n_estimators=E)
    else:
        for a in cfg["reg_aggregators"]:
            agg.get_reg_aggregator(a)
        for E in e_list:
            probs = out.probs[:E]
            rows += _rows(cfg, cell, split, spec, bname, "single", _score(split, agg.aggregate_reg("mixture", probs[:1], out.dist, levels), levels), "reference", n_estimators=E)
            for a in cfg["reg_aggregators"]:
                pred = agg.aggregate_reg(a, probs, out.dist, levels, **(cfg["aggregator_kwargs"].get(a, {}) or {}))
                rows += _rows(cfg, cell, split, spec, bname, a, _score(split, pred, levels), "baseline" if a == "mixture" else "method", n_estimators=E)
            if "native" in cfg["references"]:
                nat = ctx.native(split, E, cell.seed)
                rows += _rows(cfg, cell, split, spec, bname, "native", score_regression(split.y_test, nat["mean"], nat["quantiles"], levels), "reference", n_estimators=E)
    return rows


def run_aggregation(cfg: dict, backend: Backend | None = None, shard=None) -> Path:
    cfg = normalize_config(cfg, "aggregation")
    ctx = Context(cfg, backend)
    cells = aggregation_cells(cfg, ctx)
    tasks = {name: load_split_task(ctx, name) for name in ctx.specs}
    meta = {
        "experiment": "aggregation",
        "config": cfg,
        "backend": ctx.backend.params(),
        "tabpfn_version": _tabpfn_version_meta(ctx.backend),
        "expected": {"datasets": tasks, "methods": aggregation_methods(cfg), "seeds": cfg["seeds"]},
        "split_seed_rule": "split_seed == seed",
    }
    return _run_cells(cfg, ctx, cells, lambda c: _aggregation_cell(cfg, ctx, c), shard, meta)


def load_split_task(ctx: Context, name: str) -> str:
    return ctx.specs[name].get("task", "classification")


# --------------------------------------------------------------------------- experiment B


def coreset_grid(cfg, ctx: Context, name: str) -> tuple[int, dict[str, int], list[int]]:
    """(n_train, {budget_key: budget}, extra learning-curve budgets for `random`)."""
    split = ctx.split(name, cfg["split_seeds"][0])
    n = split.n_train
    absolute = [a for a in cfg["budget_absolute"] if 2 * a < n]
    dropped = sorted(set(cfg["budget_absolute"]) - set(absolute))
    if dropped:
        log.info("dataset %s (n_train=%d): absolute budgets %s dropped (need n_train > 2 * budget)", name, n, dropped)
    keys = budget_keys(n, cfg["budget_fractions"], absolute, split.n_classes)
    lc = [b for b in learning_curve_budgets(n, cfg["learning_curve_points"], split.n_classes) if b not in keys.values()]
    return n, keys, lc


def coreset_methods_by_task(cfg) -> dict[str, list[str]]:
    out = {}
    for task in ("classification", "regression"):
        ms = []
        for m in cfg["coreset_methods"]:
            if m == "builtin_majority_downsample" and task == "regression":
                continue
            ms.append(m)
            if m in cs.PRIOR_SHIFTING and task == "classification":
                ms.append(f"{m}_corrected")
            if m in cs.PER_ESTIMATOR_METHODS:
                ms += [f"{m}+{a}" for a in cfg["combined_aggregators"]]
        out[task] = ["full", *ms] if "full" not in ms else ms
    return out


def coreset_family(cfg, ctx) -> list[tuple[str, str, str]]:
    """Pre-registered comparisons vs `random`: (method, budget_key, metric) over all confirm datasets."""
    tasks = {d.get("task", "classification") for d in ctx.specs.values() if d.get("split") == "confirm"}
    keys = sorted({k for name, d in ctx.specs.items() if d.get("split") == "confirm" for k in coreset_grid(cfg, ctx, name)[1]})
    fam = []
    by_task = coreset_methods_by_task(cfg)
    for task in sorted(tasks):
        metrics = CLF_METRICS if task == "classification" else REG_METRICS
        for m in by_task[task]:
            if m in REFERENCE_METHODS_B:
                continue
            for k in keys:
                for metric in metrics:
                    if (m, k, metric) not in fam:
                        fam.append((m, k, metric))
    return fam


def coreset_cells(cfg, ctx: Context) -> tuple[list[Cell], dict]:
    cells: list[Cell] = []
    grids = {}
    methods = [m for m in cfg["coreset_methods"] if m not in ("full",)]
    for name in ctx.specs:
        n, keys, lc = coreset_grid(cfg, ctx, name)
        task = ctx.specs[name].get("task", "classification")
        grids[name] = {"n_train": n, "budgets": keys, "learning_curve": lc}
        for s in cfg["split_seeds"]:
            for seed in cfg["selection_seeds"]:
                cells.append(Cell("coreset", name, s, seed, "full", n, "full"))
                for key, b in keys.items():
                    for m in methods:
                        if m == "builtin_majority_downsample" and task == "regression":
                            continue
                        cells.append(Cell("coreset", name, s, seed, m, b, key))
                if "random" in methods:
                    for b in lc:
                        cells.append(Cell("coreset", name, s, seed, "random", b, "lc"))
    return cells, grids


def _stack_outputs(outs: list, task):
    if task == "classification":
        return ClfOutputs(np.concatenate([o.logits for o in outs]), outs[0].temperature, outs[0].classes, outs[0].meta)
    ref = outs[0].dist
    probs = [outs[0].probs] + [translate_probs(o.probs, o.dist.borders, ref.borders) for o in outs[1:]]
    return RegOutputs(np.concatenate(probs), ref, outs[0].y_mean, outs[0].y_std, outs[0].meta)


def evaluate_coreset(cfg, ctx: Context, split: Split, method: str, budget: int, seed: int, kwargs=None):
    """Select from the training split only, fit the backend, aggregate, score on the untouched test split.

    Returns a list of (method_name, metrics, role).
    """
    task = split.task
    X, y = split.X_train, split.y_train
    E = max(_n_estimators_list(cfg))
    levels = cfg["quantile_levels"]
    kw = dict(kwargs if kwargs is not None else (cfg["coreset_kwargs"].get(method, {}) or {}))
    aggregator = cfg["aggregator"] if task == "classification" else "mixture"
    idx = None
    if method == "full":
        out = ctx.outputs(split, X, y, E, seed, "full")
    elif method in BUILTIN_METHODS:
        sampler = BUILTIN_METHODS[method]
        if sampler not in ctx.backend.subsample_methods():
            raise ValueError(f"backend does not offer SAMPLE_SUBSAMPLING_METHOD={sampler!r}: {ctx.backend.subsample_methods()}")
        overrides = {"SUBSAMPLE_SAMPLES": int(budget), "SAMPLE_SUBSAMPLING_METHOD": sampler}
        out = ctx.outputs(split, X, y, E, seed, f"{method}:{budget}", overrides=overrides)
    elif method in cs.PER_ESTIMATOR_METHODS:
        subsets = cs.per_estimator_subsets(method, X, y, budget, E, seed, task)
        outs = [ctx.outputs(split, X[s], y[s], 1, seed * 1000 + e, f"{method}:{budget}:{e}") for e, s in enumerate(subsets)]
        out = _stack_outputs(outs, task)
    elif method == CANARY:
        idx = cs.select_coreset("random", X, y, budget, seed, task)
        y_bad = np.random.default_rng([seed, 999]).permutation(y[idx])
        out = ctx.outputs(split, X[idx], y_bad, E, seed, f"{method}:{budget}")
    else:
        if method == "embedding_kmeans":
            kw.setdefault("backend", ctx.backend)
        # the subset is a set: sorting makes a budget >= n_train selection identical to the full context
        idx = np.sort(cs.select_coreset(method, X, y, budget, seed, task, **kw))
        out = ctx.outputs(split, X[idx], y[idx], E, seed, f"{method}:{budget}:{json.dumps({k: v for k, v in kw.items() if k != 'backend'}, sort_keys=True)}")
    results = []
    if task == "classification":
        P = agg.aggregate_clf(aggregator, out.logits, out.temperature)
    else:
        P = agg.aggregate_reg(aggregator, out.probs, out.dist, levels)
    role = "reference" if method in REFERENCE_METHODS_B else "method"
    if method == "random":
        role = "baseline"
    if method == CANARY:
        role = "canary"
    results.append((method, _score(split, P, levels), role))
    if method in cs.PRIOR_SHIFTING and task == "classification" and idx is not None:
        C = split.n_classes
        Pc = cs.prior_correct(P, cs.class_prior(y[idx], C), cs.class_prior(y, C))
        results.append((f"{method}_corrected", _score(split, Pc, levels), "method"))
    if method in cs.PER_ESTIMATOR_METHODS:
        for a in cfg["combined_aggregators"]:
            if task == "classification":
                Pa = agg.aggregate_clf(a, out.logits, out.temperature, **(cfg["aggregator_kwargs"].get(a, {}) or {}))
            else:
                Pa = agg.aggregate_reg(a, out.probs, out.dist, levels)
            results.append((f"{method}+{a}", _score(split, Pa, levels), "method"))
    return results


def _coreset_cell(cfg, ctx: Context, cell: Cell) -> list[dict]:
    split = ctx.split(cell.dataset, cell.split_seed)
    spec = ctx.specs[cell.dataset]
    E = max(_n_estimators_list(cfg))
    rows = []
    for method, metrics, role in evaluate_coreset(cfg, ctx, split, cell.method, cell.budget, cell.seed):
        rows += _rows(cfg, cell, split, spec, ctx.backend.name, method, metrics, role, budget=cell.budget, budget_key=cell.budget_key, n_estimators=E)
    return rows


def run_coreset(cfg: dict, backend: Backend | None = None, shard=None) -> Path:
    cfg = normalize_config(cfg, "coreset")
    ctx = Context(cfg, backend)
    cells, grids = coreset_cells(cfg, ctx)
    family = coreset_family(cfg, ctx)  # written before any cell runs; `analyze` uses exactly this list
    meta = {
        "experiment": "coreset",
        "config": cfg,
        "backend": ctx.backend.params(),
        "tabpfn_version": _tabpfn_version_meta(ctx.backend),
        "family": family,
        "grids": grids,
        "expected": {
            "datasets": {n: d.get("task", "classification") for n, d in ctx.specs.items()},
            "methods": coreset_methods_by_task(cfg),
            "seeds": cfg["selection_seeds"],
            "budgets": {n: sorted(g["budgets"].values()) for n, g in grids.items()},
            "required_baselines": [b for b in REQUIRED_BASELINES_B if b == "full" or b in cfg["coreset_methods"]],
        },
    }
    return _run_cells(cfg, ctx, cells, lambda c: _coreset_cell(cfg, ctx, c), shard, meta)


# --------------------------------------------------------------------------- tuning (dev datasets only)


def tune_coreset(cfg: dict, method: str, param_grid: list[dict], metric: str | None = None, datasets=None, backend=None, max_seeds=3):
    """Pick coreset kwargs on `dev` datasets only. Raises TuningOnConfirmError if a confirm dataset is passed."""
    cfg = normalize_config(cfg, "coreset")
    ctx = Context(cfg, backend)
    confirm = {n for n, d in ctx.specs.items() if d.get("split") == "confirm"}
    names = list(datasets) if datasets is not None else [n for n, d in ctx.specs.items() if d.get("split") == "dev"]
    bad = sorted(set(names) & confirm)
    if bad:
        raise TuningOnConfirmError(f"tuning may only use dev datasets; got confirm datasets {bad}")
    records = []
    for params in param_grid:
        for name in names:
            task = ctx.specs[name].get("task", "classification")
            m = metric or ("log_loss" if task == "classification" else "rmse")
            _, keys, _ = coreset_grid(cfg, ctx, name)
            split = ctx.split(name, cfg["split_seeds"][0])
            for seed in cfg["selection_seeds"][:max_seeds]:
                for b in keys.values():
                    res = evaluate_coreset(cfg, ctx, split, method, b, seed, kwargs=params)
                    records.append({"params": json.dumps(params, sort_keys=True), "dataset": name, "value": res[0][1][m], "metric": m})
    table = pd.DataFrame(records)
    from .metrics import higher_is_better

    summary = table.groupby(["params", "dataset"])["value"].mean().groupby("params").mean()
    best = summary.idxmax() if higher_is_better(table["metric"].iloc[0]) else summary.idxmin()
    return json.loads(best), summary
