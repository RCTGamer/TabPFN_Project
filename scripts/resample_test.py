"""Quick check: does inflating the training split by resampling help TabPFN?

Conditions per split seed (same test rows for all):
  full   the training split as is (n rows)
  dup4   every training row `factor` times (exact copies)
  boot4  factor * n rows drawn with replacement
  boot1  n rows drawn with replacement (duplicates without growing the context)

    python scripts/resample_test.py                                # tabpfn 3.5 on phoneme, 5 seeds, E = 1 and 8
    python scripts/resample_test.py --backend sklearn --synthetic  # local smoke check, no GPU or download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_lab.aggregation import aggregate_clf
from tabpfn_lab.backends import make_backend
from tabpfn_lab.datasets import load_split, make_split
from tabpfn_lab.metrics import higher_is_better, score_classification
from tabpfn_lab.sizes import make_synthetic


def contexts(n, factor, seed) -> dict[str, np.ndarray]:
    """Row indices into the training split per condition, shuffled so copies are not adjacent."""
    rng = np.random.default_rng([seed, 808])
    idx = {
        "full": np.arange(n),
        f"dup{factor}": np.tile(np.arange(n), factor),
        f"boot{factor}": rng.integers(0, n, factor * n),
        "boot1": rng.integers(0, n, n),
    }
    return {k: rng.permutation(v) for k, v in idx.items()}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per (E, condition, metric), and the paired difference to `full` (improvement > 0 is better)."""
    rows = []
    for (E, metric), g in df.groupby(["n_estimators", "metric"]):
        wide = g.pivot_table(index="seed", columns="condition", values="value")
        sign = 1.0 if higher_is_better(metric) else -1.0
        for c in wide.columns:
            imp = sign * (wide[c] - wide["full"])
            rows.append({
                "n_estimators": E, "metric": metric, "condition": c, "n_context": int(g.loc[g["condition"] == c, "n_context"].iloc[0]),
                "mean": wide[c].mean(), "improvement_vs_full": imp.mean(), "se": imp.std(ddof=1) / np.sqrt(len(imp)) if len(imp) > 1 else np.nan,
                "seconds": g.loc[g["condition"] == c, ["fit_seconds", "predict_seconds"]].sum(axis=1).mean(),
            })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tabpfn", "sklearn"], default="tabpfn")
    ap.add_argument("--dataset-id", type=int, default=1489, help="OpenML id (1489 = phoneme)")
    ap.add_argument("--synthetic", action="store_true", help="synthetic stand-in data instead of OpenML")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--estimators", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", default="results/tables/resample_phoneme.csv")
    args = ap.parse_args(argv)

    spec = {"name": "phoneme", "source": "openml", "data_id": args.dataset_id, "task": "classification"}
    backend = make_backend({"name": "tabpfn", "version": "3.5", "device": "cuda"} if args.backend == "tabpfn" else {"name": "sklearn"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    if args.synthetic:
        Xs, ys = make_synthetic(5400, 5, n_classes=2, seed=0)

    for seed in range(args.seeds):
        split = make_split(Xs, ys, "classification", 0.3, seed, "synthetic") if args.synthetic else load_split(spec, seed)
        for cond, idx in contexts(split.n_train, args.factor, seed).items():
            for E in args.estimators:
                o = backend.clf_outputs(split.X_train[idx], split.y_train[idx], split.X_test, split.classes, E, seed)
                scores = score_classification(split.y_test, aggregate_clf("mean", o.logits, o.temperature))
                t = o.meta.get("timing", {})
                rows = [{"dataset": split.name, "seed": seed, "condition": cond, "n_estimators": E, "n_context": len(idx),
                         "n_unique": len(np.unique(idx)), "metric": m, "value": v,
                         "fit_seconds": t.get("fit_seconds", np.nan), "predict_seconds": t.get("predict_seconds", np.nan)}
                        for m, v in scores.items()]
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"seed {seed} {cond:>6} E={E} n={len(idx):>6} log_loss={scores['log_loss']:.4f} acc={scores['accuracy']:.4f}", flush=True)

    s = summarize(pd.read_csv(out))
    s.to_csv(out.with_name(out.stem + ".summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n== mean over seeds; improvement_vs_full > 0 means better than the plain training set ==")
    print(s.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
