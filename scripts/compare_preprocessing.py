"""Compare TabPFN configurations on one dataset and print a results table.

    python scripts/compare_preprocessing.py --n-estimators 4 --max-rows 2000

Add your own configurations to ``CONFIGS`` below.
"""

from __future__ import annotations

import argparse
import csv
import functools
from pathlib import Path

import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tabpfn import TabPFNClassifier

from tabpfn_lab.benchmark import run_benchmark
from tabpfn_lab.preprocessing import (
    CastFloat32,
    DropConstantColumns,
    DropCorrelatedColumns,
    deduplicate_rows,
    stratified_subsample,
)


def build_configs(args, device):
    def tabpfn(**kw):
        return TabPFNClassifier(device=device, n_estimators=args.n_estimators, random_state=0, **kw)

    subsample = functools.partial(stratified_subsample, max_rows=args.max_rows)
    column_steps = [DropConstantColumns(), DropCorrelatedColumns(0.99), CastFloat32()]

    # name -> (estimator, row_reducers)
    return {
        "baseline": (tabpfn(), []),
        "columns": (make_pipeline(*column_steps, tabpfn()), []),
        "dedup": (tabpfn(), [deduplicate_rows]),
        "subsample": (tabpfn(), [subsample]),
        "all": (make_pipeline(*column_steps, tabpfn()), [deduplicate_rows, subsample]),
        # TabPFN's built-in levers, for reference against the custom steps.
        "low_memory": (tabpfn(fit_mode="low_memory"), []),
        "fit_with_cache": (tabpfn(fit_mode="fit_with_cache"), []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openml", default="credit-g", help="OpenML dataset name")
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results/compare.csv"))
    args = parser.parse_args()

    X, y = fetch_openml(args.openml, version=1, as_frame=True, return_X_y=True)
    X = X.apply(lambda c: c.cat.codes if c.dtype.name == "category" else c).to_numpy(float)
    y = y.astype("category").cat.codes.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for repeat in range(args.repeats):
        # Rebuild each time so no config reuses a previous fit.
        for name, (est, reducers) in build_configs(args, device).items():
            r = run_benchmark(name, est, X_train, y_train, X_test, y_test, reducers)
            rows.append({"repeat": repeat, **r.as_dict()})
            print(rows[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
