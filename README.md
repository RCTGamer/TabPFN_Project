# tabpfn-lab

A research harness for two experiments on TabPFN-3.5 (or any similar tabular foundation model):

- **Experiment A (aggregation):** replace the plain mean over ensemble-member outputs with other aggregation rules.
- **Experiment B (coresets):** shrink the training context to a fixed row budget and try to beat uniform random subsampling.

The full design is in [`PROJECT_SPEC.md`](PROJECT_SPEC.md). **Research only:** the TabPFN-3.5 open weights are under a non-commercial licence.

## Install

```bash
pip install -e ".[dev]"            # laptop / CI: sklearn backend only
pip install -e ".[dev,tabpfn]"     # GPU server: adds tabpfn==9.0.0 and torch
```

## Quickstart

```bash
pytest                                  # fast, sklearn backend (~2 min; GPU tests skip without tabpfn + CUDA)
pytest -m "not slow"                    # skip the simulation-heavy statistical tests
pytest -m tabpfn                        # real model, on the GPU box (RUN_TABPFN=0 disables)
RUN_TIER_LARGE=1 pytest -m tier_large   # opt-in: a real n_train >= 100k dataset

python -m tabpfn_lab.cli aggregation --config configs/smoke.yaml   # ~10 s sanity run
python -m tabpfn_lab.cli coreset     --config configs/smoke.yaml   # ~20 s
python -m tabpfn_lab.cli analyze     --csv results/tables/smoke_coreset.csv
```

Run the smoke config after every code change before launching a long GPU run.

## GPU server runs

```bash
export TABPFN_TOKEN=...   # one-time licence acceptance; see "GPU server setup" below
mkdir -p results/logs

# plan first: refuses (exit 2) when power for min_detectable_effect is below 0.8, unless --force
python -m tabpfn_lab.cli plan --config configs/coreset.yaml [--pilot results/tables/pilot.csv]

# one shard per GPU, safe under nohup/tmux; a killed run resumes where it stopped
CUDA_VISIBLE_DEVICES=0 nohup python -m tabpfn_lab.cli coreset --config configs/coreset.yaml --shard 0/2 > results/logs/c0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m tabpfn_lab.cli coreset --config configs/coreset.yaml --shard 1/2 > results/logs/c1.out 2>&1 &

python -m tabpfn_lab.cli merge --name coreset_v1      # concatenates shards and runs validate_results
python -m tabpfn_lab.cli analyze --csv results/tables/coreset_v1.csv
```

What the runner gives you:

- **Cache:** per-estimator outputs go to `results/raw/` as npz, keyed by dataset, split, seed, backend settings, estimator count and subset. Re-scoring with a new aggregator never calls the backend.
- **Resume:** rows are appended after each cell, and cells already in the CSV are skipped.
- **Failures are data:** a CUDA OOM or any other exception inside a cell goes to `<name>.failures.csv` and the sweep continues. `validate` reports failed and missing cells.
- **Logs:** `results/logs/<name>.log` gets one line per cell, with wall time and peak GPU memory.
- **Metadata:** `<name>.meta.json` holds the config, backend parameters, `tabpfn` version, the resolved inference config, test-split hashes, the expected grid, and the pre-registered Holm family for experiment B. The family is written before the first cell runs.

### GPU server setup

TabPFN 9.0.0 needs a one-time licence acceptance before it downloads weights. Log in at the Prior Labs portal, accept the licence, then `export TABPFN_TOKEN=<api key>`. For gated Hugging Face mirrors, run `hf auth login` or set `HF_TOKEN`. The backend raises a clear error naming these variables when the download is refused.

## Layout

| module | contents |
|---|---|
| `backends.py` | `Backend` ABC, `SklearnBackend` (random forests, for tests), `TabPFNBackend`, `align_logits`/`align_proba`, numpy `BarDistribution`. **The only module that imports `tabpfn`/`torch`.** |
| `aggregation.py` | classification aggregators (`CLF_AGGREGATORS`) and regression aggregators (`REG_AGGREGATORS`) |
| `coresets.py` | single-subset selection (`CORESET_METHODS`), per-estimator subsets (`PER_ESTIMATOR_METHODS`), prior correction |
| `experiments.py` | `run_aggregation`, `run_coreset`, `tune_coreset` (dev only), caching/resume/sharding |
| `metrics.py`, `stats.py`, `effect.py`, `sizes.py`, `validate.py` | scoring, dataset-level paired statistics and Holm, effect sizes and power, size tiers and budgets, schema validation |
| `estimator.py` | `TabPFNLabClassifier` / `TabPFNLabRegressor`, scikit-learn estimators for external benchmarks |
| `cli.py` | `aggregation`, `coreset`, `analyze`, `merge`, `validate`, `plan` |

## Adding a method

**Aggregator:** write `def agg_foo(logits, temperature, **kw) -> (n, C) probabilities` in `aggregation.py` (apply `temperature` to each estimator first), then add `"foo": agg_foo` to `CLF_AGGREGATORS`. If it needs a train-only validation slice, add the name to `NEEDS_VALIDATION` and accept `val_logits`/`val_y`. For regression: `def f(probs, dist, levels, **kw) -> RegPrediction`, registered in `REG_AGGREGATORS`.

**Coreset method:** write `def sel_foo(X, y, budget, rng, task, seed, **kw) -> indices` in `coresets.py`, then add `"foo": sel_foo` to `CORESET_METHODS`. Bound any expensive step by `budget`/`pilot`, never by `n`. Classification methods must cover every class when `budget >= n_classes`; `_ensure_class_coverage` does that. If the method changes class proportions, add it to `PRIOR_SHIFTING`, and the runner will also emit a `<name>_corrected` row. Method-specific settings go in the config under `coreset_kwargs: {foo: {...}}`.

`pytest tests/test_aggregation.py tests/test_coresets.py tests/test_sizes.py` checks the new method against the contract automatically through the registries.

## Results schema

Long format, one row per measurement: `experiment, dataset, split (dev|confirm), tier, task, seed, method, budget, ratio, n_train, metric, value, backend, role, flags`. The runner adds four columns: `split_seed`, `n_estimators`, `budget_key` (`f0.1`, `a500`, `lc` for learning-curve budgets, `full`), and `cell`. `validate_results` checks the schema, value ranges, duplicates, grid completeness, failed cells, the baselines required at every coreset cell (`full`, `random`, `builtin_subsample`, `builtin_majority_downsample`), and the six or more `random` learning-curve budgets per dataset.

## Notes from reading the tabpfn 9.0.0 source

- `translate_probs_across_borders` takes **logits** and applies the softmax itself. `TabPFNRegressor._iter_forward_executor` already applies the per-estimator softmax temperature and the target-transform border mapping. The regression capture loop in `TabPFNBackend._capture_reg` follows `_compute_aggregated_logits` without the averaging. Point predictions decode through `raw_space_bardist_`, a `FullSupportBarDistribution` with half-normal tails. `backends.BarDistribution(full_support=True)` reproduces its `mean`, `icdf` and NLL in numpy, so offline scoring needs no torch. `test_baseline_reproduces_native_reg` guards all of this.
- **`tabpfn.downsample_correction` does not exist in 9.0.0.** Prior correction for prior-shifting coresets is the harness's own Bayes re-weighting, `coresets.prior_correct`.
- `SampleSubsamplingMethod` has the values `auto`, `balanced`, `stratified` and `majority_downsample`. The backend reads them at runtime. `majority_downsample` raises when the budget is at or below the number of non-majority rows. Those cells land in the failures CSV. `SUBSAMPLE_SAMPLES` also accepts explicit per-estimator index lists.
- OOM: `tabpfn.errors.TabPFNCUDAOutOfMemoryError` / `TabPFNOutOfMemoryError` wrap `torch.OutOfMemoryError`. `backends.is_oom_error` recognises both.
- Checked from source but **not yet run on a GPU**: the resolved 3.5 inference config, the checkpoint's `N_ESTIMATORS`/`SOFTMAX_TEMPERATURE`, mixed-checkpoint ensembles, and the tolerance for the native-regression reproduction test. The first `pytest -m tabpfn` run on the server settles these; the resolved config is saved in every run's `meta.json`.
