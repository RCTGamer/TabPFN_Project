# tabpfn-lab: project spec (for Claude Code)

Build this repository from scratch. It is a research harness for two experiments on TabPFN-3.5 (or any similar tabular foundation model):

- **Experiment A, aggregation:** replace the plain mean over estimator outputs with other aggregation rules.
- **Experiment B, coresets:** shrink the training context to a fixed row budget and beat uniform random subsampling.

Work in the order given in section 8. Do not write code beyond what this spec asks for.

---

## 1. Ground rules

1. **Backend-agnostic.** Experiments never import `tabpfn` directly. They call a `Backend` interface (section 3.1). Real experiments run on a GPU server over SSH, so every experiment config defaults to `TabPFNBackend` and GPU cost is not a design constraint. A cheap `SklearnBackend` (random forests) still exists, but only so the unit tests finish in seconds and test the pipeline logic independently of model weights, downloads, or GPU nondeterminism.
2. **Per-estimator outputs are the unit of caching.** Run the expensive model once per (dataset, split, seed, backend config), save the raw per-estimator outputs, then evaluate every aggregator offline for free.
3. **No leakage, ever.** Anything that selects rows or fits weights may only see the training split. Tests enforce this.
4. **Everything is seeded.** Same config + same seed gives bit-identical results with the sklearn backend.
5. **Resumable.** A crashed sweep restarts and skips finished cells.
6. **Results are long-format DataFrames/CSVs** with a validated schema (section 3.5).
7. **Pin the package.** The behaviour described in section 3.1 was read from `tabpfn` 9.0.0 source (`base.py`, `classifier.py`, `regressor.py`). Pin `tabpfn==9.0.0`, record `tabpfn.__version__` in every results file, and re-run the baseline-reproduction tests (7.9) after any upgrade, because the regression capture code uses private methods.
8. **Seeds map to `random_state`.** `TabPFNClassifier/Regressor` default to `random_state=0`, so omitting it makes every "seed" identical. Always pass the experiment seed explicitly.

---

## 2. Repository layout

```
tabpfn-lab/
├── pyproject.toml            # deps: numpy pandas scikit-learn scipy pyyaml; extras: tabpfn(torch), dev(pytest)
├── README.md                 # quickstart: smoke run, full run, how to add an aggregator/coreset method
├── CLAUDE.md                 # short: conventions, how to run tests, never import tabpfn outside backends.py
├── configs/
│   ├── smoke.yaml            # tiny synthetic data, sklearn backend, runs in <1 min (used by CI + tests)
│   ├── aggregation.yaml      # real datasets, tabpfn backend
│   └── coreset.yaml
├── src/tabpfn_lab/
│   ├── backends.py           # Backend ABC, SklearnBackend, TabPFNBackend, align_proba
│   ├── aggregation.py        # classification + regression aggregators, REGISTRY
│   ├── coresets.py           # selection methods, per-estimator subset generators, REGISTRY
│   ├── metrics.py            # log_loss, accuracy, roc_auc, ece, brier, rmse, nll, crps_from_quantiles, coverage
│   ├── datasets.py           # loaders (OpenML + synthetic), Split dataclass, make_split
│   ├── cache.py              # OutputCache: keyed npz store for per-estimator outputs
│   ├── experiments.py        # run_aggregation(), run_coreset(), shared scoring helpers
│   ├── sizes.py              # size tiers, budget resolution, synthetic generators with known learning curves
│   ├── effect.py             # equivalent_budget, gap_closed, headroom flags, power / min-detectable-effect
│   ├── stats.py              # paired bootstrap CI, sign test, Wilcoxon, Holm correction, win rate (dataset-level weighting)
│   ├── estimator.py          # sklearn-compatible estimator wrapping backend + aggregator + coreset (for external benchmarks)
│   ├── validate.py           # validate_results(df, expected grid) -> list[str] of problems
│   └── cli.py                # `python -m tabpfn_lab.cli {aggregation|coreset|analyze} --config ...`
├── tests/                    # section 7
│   ├── conftest.py
│   ├── test_aggregation.py
│   ├── test_coresets.py
│   ├── test_metrics.py
│   ├── test_stats.py
│   ├── test_cache.py
│   ├── test_protocol.py
│   ├── test_validate.py
│   ├── test_sizes.py
│   ├── test_effect_sizes.py
│   ├── test_design.py
│   ├── test_estimator.py
│   ├── test_experiment_sanity.py
│   └── test_tabpfn_integration.py
└── results/                  # gitignored; raw/ (cached outputs), tables/ (CSVs)
```

---

## 3. Core interfaces

### 3.1 Backend (`backends.py`)

**What the package actually does (read from tabpfn 9.0.0 source).** One `fit` with `n_estimators=E` builds E ensemble members. They differ by preprocessing, feature shifts, class permutations (classification), target transforms (regression), feature subsampling, and optionally row subsampling. Per-estimator outputs are therefore readable from a single fit, and **no "E separate calls" proxy is needed**.

- *Classification.* `clf.predict_raw_logits(X)` returns `(E, n, C)` raw logits with no temperature and no averaging. Class permutations are already undone, so columns follow `clf.classes_`. The package's default probability is `clf.logits_to_probabilities(raw)`: divide by temperature `T`, softmax per estimator, then arithmetic mean over estimators (with the default `average_before_softmax=False`). `T` is `clf.softmax_temperature_`, taken from the checkpoint (the docstring says 0.9 for releases up to v8.5.0; read the real value for 3.5, do not assume).
- *Regression.* Each estimator emits logits over its own target-transform-dependent bin borders. The package applies the per-estimator temperature, maps each distribution onto the shared z-normalised borders (`translate_probs_across_borders`), averages probabilities across estimators (a mixture; with `average_before_softmax=True` it averages log-probabilities instead), takes the log, applies an optional calibrated ensemble temperature, then decodes mean/median/quantiles through `znorm_space_bardist_` and un-normalises with `* y_std + y_mean`. There is no public per-estimator accessor, so the backend captures the per-estimator distributions by replicating the loop in `TabPFNRegressor._compute_aggregated_logits` (uses `_iter_forward_executor` and `translate_probs_across_borders`). Keep the copy minimal and covered by the reproduction test in 7.9.

```python
@dataclass
class ClfOutputs:
    logits: np.ndarray        # (E, n_test, C) raw logits, columns ordered like `classes`
    temperature: float        # per-estimator softmax temperature the package would apply
    classes: np.ndarray
    meta: dict                # tabpfn version, resolved InferenceConfig, n_estimators_, device, precision

@dataclass
class RegOutputs:
    probs: np.ndarray         # (E, n_test, B) per-estimator probabilities on shared z-normalised bins
    dist: "BinDistribution"   # mean(log_probs), icdf(log_probs, q), nll(log_probs, y) in raw units
    y_mean: float
    y_std: float
    meta: dict

class Backend(ABC):
    name: str
    def clf_outputs(self, X_train, y_train, X_test, classes, n_estimators, seed) -> ClfOutputs
    def reg_outputs(self, X_train, y_train, X_test, n_estimators, seed) -> RegOutputs
    def native_clf(self, X_train, y_train, X_test, classes, n_estimators, seed) -> np.ndarray   # (n, C) = clf.predict_proba
    def native_reg(self, X_train, y_train, X_test, n_estimators, seed, levels) -> dict           # mean, median, quantiles from reg.predict
```

- `native_*` run the package's own prediction path in one call. They are the `native` reference row in every experiment and the ground truth that the offline `mean`/`mixture` aggregators must reproduce (test 7.9).
- `align_proba` / `align_logits(logits, model_classes, classes)`: place outputs into the full class ordering. Needed because a coreset can miss a class; missing classes get a very negative logit.
- `BinDistribution` is a tiny protocol (`mean`, `icdf`, `nll`). For `TabPFNBackend` wrap `znorm_space_bardist_` / `raw_space_bardist_` (check which of `mean`, `icdf`, and a log-prob/NLL method exist on `FullSupportBarDistribution`). For `SklearnBackend` implement it with a histogram.
- `SklearnBackend`: estimator `k` is a `RandomForest*` with `random_state = seed*1000 + k`. Classification logits are `log(proba)`. Regression builds fixed bins from training-target quantiles and histograms the per-tree predictions.
- `TabPFNBackend(version, device, fit_mode="fit_preprocessors", inference_precision="auto", **extra)`:
  - Build with `TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, n_estimators=E, random_state=seed, device=device, **extra)`. `ModelVersion` lives in `tabpfn.constants` and has `V3`, `V3_5`, `V3_5_FAST` (also V2, V2_5, V2_6). Same for the regressor.
  - Always pass `n_estimators=E` as an int. With `"auto"` the count comes from the checkpoint and may be raised for wide data; record `n_estimators_` after fit.
  - On first fit, save `clf.get_inference_config()` (as a dict) into `meta`. This shows what preprocessing 3.5 really uses (e.g. `PREPROCESS_TRANSFORMS`, `REGRESSION_Y_PREPROCESS_TRANSFORMS`, `N_ESTIMATORS`, `SOFTMAX_TEMPERATURE`).
  - Use `fit_mode="fit_preprocessors"` for everything (it is the default). `fit_with_cache` is only for latency measurements.
  - Optional `inference_precision=torch.float32` for more reproducible numbers at some speed cost.
  - `model_path` also accepts a list of checkpoints applied across estimators; this enables an optional mixed-checkpoint ensemble experiment (see 9).

### 3.2 Aggregators (`aggregation.py`)

Classification aggregators take raw logits `L: (E, n, C)` and temperature `T`, and return `(n, C)` valid probabilities. All of them apply `T` first, as the package does:

| name | rule |
|---|---|
| `mean` | `mean_e softmax(L_e / T)`. **This is the package default** and the baseline. |
| `logit_mean` | `softmax(mean_e L_e / T)`. Equivalent to the package's built-in `average_before_softmax=True` (normalised geometric mean); kept as a second reference, not a novel method. |
| `median` | per-class median of per-estimator probabilities, renormalise |
| `trimmed_mean` | `scipy.stats.trim_mean` along the estimator axis (param `trim`), renormalise |
| `entropy_weighted` | per-sample weights `softmax(-H_e / tau)` across estimators |
| `weighted` | fixed estimator weights `w` (sum to 1) |
| `val_weighted` | weights from held-out context rows: `w_e ∝ exp(-logloss_e / tau)`; fit on a train-only validation slice |
| `mean_temp_offline` | `mean`, then a temperature fit on a train-only validation slice. The package already has a built-in version (`tuning_config={"calibrate_temperature": True}`, with `eval_metric`); include that as the `native_tuned` reference row and treat this one as a re-implementation to compare against it. |

Regression aggregators take per-estimator bin probabilities `B: (E, n, bins)` on the shared z-normalised borders, and return aggregated log-probabilities `(n, bins)`. Point predictions, quantiles, CRPS and NLL are computed from them through `RegOutputs.dist`:

| name | rule |
|---|---|
| `mixture` | mean of per-estimator probabilities, then log. **Package default** and the baseline. |
| `log_pool` | mean of per-estimator log-probabilities, renormalise. Equivalent to the package's `average_before_softmax=True`. |
| `quantile_mean` | decode each estimator's quantiles via `dist.icdf`, average each level across estimators (Vincentization), then score from quantiles |
| `quantile_median` | median across estimators per level |

Registries: `CLF_AGGREGATORS`, `REG_AGGREGATORS` (name to callable). Adding a new method = one function + one registry line. Helper `point_from_log_probs(dist, logp)` returns the predictive mean.

### 3.3 Coresets (`coresets.py`)

```python
select_coreset(method, X, y, budget, seed, task="classification", **kw) -> np.ndarray  # indices into the arrays passed in
per_estimator_subsets(method, X, y, budget, n_estimators, seed, task) -> list[np.ndarray]
```

Single-subset methods: `random`, `stratified` (class-proportional for classification; target-quantile bins for regression), `balanced` (equal per class, flagged as prior-shifting), `kmeans_stratified` (per-class k-means, pick the real point nearest each centroid), `uncertainty_mix` (fraction `frac_hard` of the budget from highest-entropy / highest-tree-variance points under a small pilot model fit on a random pilot subset, the rest random).

Per-estimator methods (each estimator gets a different context; ties A and B together): `perest_random` (independent random subsets), `perest_partition` (permute once, give each estimator a cyclic window so the union covers the training set when `E*budget >= n`).

**The package already does per-estimator row subsampling.** `inference_config={"SUBSAMPLE_SAMPLES": k}` gives each estimator its own row subset, `SAMPLE_SUBSAMPLING_METHOD` selects the sampler (it includes `"majority_downsample"`), and `tabpfn.downsample_correction` (`context_class_prior`, `downsample_class_weights`, `apply_class_weights`, and for regression `downsample_bucket_log_weights`) corrects the class-prior shift this causes. Consequences:

- Experiment B must include these as baselines: `builtin_subsample` (the package's default sampler at `SUBSAMPLE_SAMPLES=budget`) and `builtin_majority_downsample`. Read the allowed `SampleSubsamplingMethod` values from the installed package at runtime rather than hard-coding them. Beating `random` alone is not enough.
- For methods that change class proportions (`balanced`, `uncertainty_mix`), report both uncorrected and prior-corrected results, reusing the package's correction functions (verify their exact semantics first, see 9).

Additional method: `embedding_kmeans`. Fit the backend on a small random pilot subset (size `pilot`), call `get_embeddings(X_train, data_source="test")` to embed every training row (shape `(E, n, dim)`; average over E; chunk the rows to bound memory), run k-means or greedy k-center in that space, and return the real rows nearest each centre. Selection stays train-only.

Contract for all: returns unique, in-range, int indices, `len == min(budget, n)`, deterministic given `seed`, and for classification with `budget >= n_classes` every class present (except `random`, which makes no such promise).

### 3.4 Experiments (`experiments.py`)

**A. `run_aggregation(cfg)`**
for dataset in datasets, for seed in seeds: split → get per-estimator outputs (from cache or backend) → for each aggregator, score. Also emit `single` (estimator 0 only) as a lower reference, `mean`/`mixture` as the baseline, `native` (the package's own prediction path) and `native_tuned` (classification only, package calibration) as references. Record the actual `n_estimators_` and the resolved inference config in the metadata. Metrics: classification `log_loss, accuracy, roc_auc, ece`; regression `rmse, crps`.

**B. `run_coreset(cfg)`**
for dataset, seed, budget, method: select subset **from the training split only**, run backend on it, aggregate with `mean`, score on the untouched test split. Always include `full` (budget = n_train) and `random` as reference rows. Per-estimator coreset methods (`perest_*`) fit the backend once per estimator with `n_estimators=1` on that estimator's own subset, then aggregate with `mean` (this differs from the package's built-in `SUBSAMPLE_SAMPLES`, which is a separate baseline). Also emit the `builtin_*` baselines from 3.3.

**C. Combined (optional, last):** `perest_*` coresets x best aggregator from A.

### 3.5 Result schema (long format, one row per measurement)

`experiment, dataset, split, tier, task, seed, method, budget, ratio, n_train, metric, value, backend, role, flags`
(`budget` and `ratio` are `NaN` for experiment A; `ratio = budget / n_train`; `split` is `dev` or `confirm`; `tier` comes from `tier_of(n_train)`; `seed` is the selection seed and the split seed is stored in the run metadata; `flags` holds e.g. `saturated`, `above_range`, `below_range`.) `validate.py` checks this schema (section 7.8).

### 3.6 Cache (`cache.py`)

`OutputCache(root)` stores `npz` files keyed by a hash of `(dataset, task, split_seed, n_train, n_test, backend name+kwargs, n_estimators, subset_id)`. API: `get(key)`, `put(key, arrays)`, `exists(key)`, `make_key(...)`. Writes are atomic (temp file + rename). Aggregation sweeps must never call the backend if the key exists.

### 3.7 Size tiers and budgets (`sizes.py`)

Dataset size is a first-class experimental factor, not a detail.

```python
TIERS = {"tiny": (200, 1_000), "small": (1_000, 10_000), "medium": (10_000, 100_000), "large": (100_000, 1_000_000)}  # n_train ranges
REDUCTION_RATIOS = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]    # budget / n_train

def tier_of(n_train) -> str
def resolve_budgets(n_train, fractions=None, absolute=None, n_classes=None) -> list[int]
    # union of round(f * n_train) and absolute budgets; unique, sorted, >= max(n_classes, 2),
    # budgets >= n_train are dropped (that cell is the `full` reference), tiny budgets are clipped up and logged.
def make_synthetic(n, n_features, n_classes=2, task="classification", imbalance=None, noise=..., seed=0)
    # fast generator whose accuracy-vs-training-size curve is known (power law m(b) = m_inf + a * b**-alpha),
    # so the effect-size code can be tested against ground truth.
```

**The experiment-2 question** is: at a given reduction ratio (e.g. 1% of a large dataset), does a chosen context beat a random context of the same size by a margin larger than noise, and how many random rows would it take to match it? Two headline numbers, both computed per (dataset, budget) and then aggregated at the dataset level:

- **`equivalent_budget_multiplier`:** (random budget needed to match the coreset's metric) / (coreset budget). Obtained by inverting the monotone-smoothed random-subset learning curve measured at several budgets. A multiplier of 3 means the coreset of 1,000 rows is as good as 3,000 random rows.
- **`gap_closed`:** (metric_random - metric_coreset) / (metric_random - metric_full), for lower-is-better metrics (sign flipped otherwise). 0 means no better than random, 1 means as good as the full data.

**Headroom guard:** if the random subset is already within noise of `full` at a budget (difference below one standard error), that cell is flagged `saturated`. Saturated cells never count towards a significance claim, because there is nothing to gain. Choose budgets where random is clearly worse than full; a short learning curve on the development datasets decides this.

**Confirmatory discipline (the way to get results that mean something):** datasets are split into a `dev` list and a `confirm` list in the config. All tuning (pilot size, `frac_hard`, k, thresholds) uses `dev` only. Claims are made only on `confirm`, using Holm-corrected tests over every (method, budget, metric) comparison in the family.

### 3.8 Benchmark-ready estimator (`estimator.py`)

Later, the modified models must run through external benchmarks (TabArena, BeyondArena, TALENT, or others) without changing experiment code. Those benchmarks run their own splits, metrics and fit/predict loops, so the only thing to build now is a drop-in estimator:

```python
class TabPFNLabClassifier(BaseEstimator, ClassifierMixin):   # and TabPFNLabRegressor
    def __init__(self, version="3.5", n_estimators=8, aggregator="mean", aggregator_kwargs=None,
                 coreset="none", budget=None, budget_fraction=None, coreset_kwargs=None,
                 random_state=0, device="cuda", **tabpfn_kwargs): ...
    def fit(self, X, y): ...          # coreset selection here, from the training data only
    def predict_proba(self, X): ...   # one fit -> per-estimator outputs -> chosen aggregator
    def predict(self, X): ...
```

- It is a thin shell over the existing `Backend`, `aggregation.py` and `coresets.py`; no logic is duplicated, so every benchmark run uses exactly the code the unit tests cover.
- Anything an aggregator needs (for example a validation slice for `val_weighted`) is carved out of the training data inside `fit`, so it can never see benchmark test rows.
- Handles pandas input, categoricals, NaNs, string labels and non-contiguous classes like the underlying package does, and follows scikit-learn conventions (`get_params`/`set_params`, `clone`).
- With `aggregator="mean"` and `coreset="none"` it must reproduce stock `TabPFNClassifier` predictions (this is the regression guard for every benchmark comparison).
- Each benchmark has its own way of registering a custom model; read that benchmark's documentation when the time comes, and keep any adapter code in a separate `benchmarks/` folder, not in the core package.

**Rules for benchmark use** (these keep later claims valid):
- Run the **unmodified TabPFN-3.5 in the same harness, on the same hardware and splits**, as the baseline. Do not compare against numbers copied from the report, because protocol, hardware and time measurement differ (for example, the 3.5 report times TabArena with 8-fold bagging plus refit).
- Benchmark datasets belong in the `confirm` set. Do not tune coreset or aggregator settings on them, otherwise the dev/confirm split in 3.7 is meaningless.
- The open weights are under a non-commercial licence; benchmarking for research is fine, but do not use the results or model outputs for commercial decisions or procurement comparisons.

---

## 4. Running experiments efficiently on the SSH GPU server

GPU cost is not a constraint, so the goal here is a statistically strong, restartable, well-logged sweep rather than a cheap one.

- **Offline aggregator sweeps:** cache the raw per-estimator outputs once (`(E, n_test, C)` logits for classification, `(E, n_test, bins)` probabilities plus borders and `y_mean/y_std` for regression); trying a new aggregator afterwards costs milliseconds. Also cache a train-only validation-slice output when an aggregator needs one (`val_weighted`, `mean_temp_offline`).
- **Long jobs over SSH:** the CLI must work under `tmux`/`nohup`: line-buffered logging to `results/logs/<name>.log`, one log line per cell with wall time and peak GPU memory, and no interactive prompts.
- **Multi-GPU sharding:** `--shard i/N` runs every N-th cell (stable ordering of the cell grid), so several processes can each pin a GPU with `CUDA_VISIBLE_DEVICES`. Shards write separate CSVs (`<name>.shard<i>.csv`); `cli merge` concatenates them and runs `validate_results`.
- **Resume:** the runner appends rows after each cell and skips cells already present, including after a crash or a killed SSH session.
- **Failures are data, not crashes:** CUDA OOM or any per-cell exception is caught, logged with the cell id to `results/tables/<name>.failures.csv`, and the sweep continues. `validate_results` reports failed cells so they are never silently missing.
- **Smoke config:** `smoke.yaml` (synthetic data, 300-600 rows, 4 estimators, 2 seeds, sklearn backend) is for `pytest` and for a 1-minute sanity check after any code change, before launching a long GPU run.
- **Dataset-level aggregation in analysis:** average over seeds within a dataset first, then compare across datasets (every dataset has equal weight, as in the TabPFN-3.5 report).
- **Statistical power (use the spare compute):** default to 10 seeds per (dataset, setting) rather than 3-5. More seeds are cheaper than ambiguous results.
- **Budget grid default:** define budgets as fractions of `n_train` (`REDUCTION_RATIOS`, section 3.7) so different dataset sizes are comparable, plus a few absolute budgets (`500, 2000, 8000, 32000`) for the same-budget-across-datasets view; always include `full`. Only use datasets with `n_train > 2 * max(budget)` for B. Large-tier runs (`n_train >= 100k`) are where context reduction matters most and where `full` is still affordable on your GPUs.
- **Learning-curve sweep for random subsets:** to compute `equivalent_budget_multiplier`, run `random` at at least 6 log-spaced budgets per dataset (add these to the grid automatically).
- **Estimator count:** run 1, 4 and 8 estimators so the ensembling headroom is visible. Since one fit now yields all E per-estimator outputs, the 1- and 4-estimator rows can be derived from the 8-estimator run by subsetting estimators 0..k-1 (the 3.5 report lists 8 as the default for 3.5 and 4 for Fast; confirm against the resolved config).
- **Batched inference (optional speed-up):** `TabPFNClassifier.predict_proba_batched` and `TabPFNRegressor.predict_batched` fuse several same-shape datasets into one pass. Useful for many seeds of one dataset at one budget. Requirements: same raw array shapes, same class set across datasets, float32, no `tuning_config`, no `majority_downsample`. Treat as an opt-in flag and verify equality with the unbatched path.
- **Reproducibility:** pass `random_state=seed`, optionally `inference_precision=torch.float32`; expect small GPU nondeterminism and set test tolerances accordingly.

---

## 5. Configs (shape)

```yaml
name: aggregation_v1
backend: {name: tabpfn, version: "3.5", device: cuda, fit_mode: fit_preprocessors}   # version: 3 | 3.5 | 3.5-fast   # smoke.yaml uses name: sklearn
datasets: [{name: "...", source: openml|synthetic, task: classification}]
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
test_size: 0.3
n_estimators: 8
aggregators: [mean, logit_mean, median, trimmed_mean, entropy_weighted, val_weighted, mean_temp_offline]
references: [native, native_tuned]   # package's own prediction path
cache_dir: results/raw
out_csv: results/tables/aggregation_v1.csv
```

Coreset config adds:

```yaml
datasets:
  dev:     [{name: "...", tier: large}, {name: "...", tier: medium}]      # tuning allowed here only
  confirm: [{name: "...", tier: large}, {name: "...", tier: medium}]      # claims made here only
budget_fractions: [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
budget_absolute: [500, 2000, 8000, 32000]
coreset_methods: [random, stratified, kmeans_stratified, uncertainty_mix, embedding_kmeans,
                  perest_random, perest_partition, builtin_subsample, builtin_majority_downsample]
selection_seeds: 10          # selection randomness; split held fixed per (dataset, split_seed)
split_seeds: 3
min_detectable_effect: 0.01  # relative metric improvement the design must be able to detect
alpha: 0.05
```

Per-tier configs (`coreset_tiny.yaml` ... `coreset_large.yaml`) only change the dataset lists and fractions, so one tier can be run alone.

---

## 6. Analysis (`stats.py`, `cli analyze`)

- `compare(df, method, baseline, metric)` aligns per (dataset, seed), averages seeds within dataset, returns mean paired difference, bootstrap CI over datasets, sign-test p-value, win rate (ties count 0.5).
- Print one table per experiment: method vs baseline, per metric, with CI. Flag "significant" only if the CI excludes 0 **and** the sign-test p < 0.05.
- Never report a method as better from a single seed or a single dataset.
- **Seeds are averaged within a dataset first; the bootstrap and tests resample datasets**, not seeds. Extra seeds shrink within-dataset noise but cannot beat the dataset-level limit, so claims need enough datasets.
- Apply **Holm correction** across the whole family of (method, budget, metric) comparisons on the `confirm` datasets; report adjusted p-values.
- Report effect sizes, not just p-values: `equivalent_budget_multiplier` and `gap_closed` (section 3.7) with bootstrap CIs, plus the raw metric difference.
- **Per-tier breakdown:** one table per size tier and per reduction ratio (a curve of `gap_closed` against ratio), with every dataset weighted equally; `saturated` cells are listed but excluded from claims.
- `cli plan --config ...`: before running, estimate power from a pilot (variance across datasets and seeds) and print the minimum detectable effect for the planned design. Refuse a confirmatory run whose power for `min_detectable_effect` is below 0.8 unless `--force` is given.

---

## 7. Unit tests

Fixtures in `conftest.py`: tiny synthetic classification split (400 rows, 3 classes, 6 features), tiny regression split, a `SklearnBackend(n_trees=10)`, a `tmp_path` cache, and a helper to build per-estimator probability tensors with known structure. Mark real-model tests `@pytest.mark.tabpfn` (they run on the GPU server; skipped automatically where `tabpfn` or CUDA is unavailable). All other tests use the sklearn backend so `pytest` stays fast and deterministic. Mark statistical tests `@pytest.mark.slow`. Size-tier tests use scaled-down stand-ins with the sklearn backend (`n` of about 300, 3,000 and 30,000 for tiny, small, medium); only `@pytest.mark.tier_large` tests (real `n >= 100k`, GPU server) touch large data, and they are opt-in.

### 7.1 `test_aggregation.py` (properties of the aggregators)
- **valid_distribution:** every classification aggregator returns finite values, `>= 0`, rows summing to 1 (atol 1e-6), shape `(n, C)`.
- **single_estimator_identity:** with `E=1`, every aggregator returns that estimator's probabilities (up to renormalisation tolerance).
- **identical_estimators:** `E` copies of the same `P` give back `P` for `mean, logit_mean, median, trimmed_mean, entropy_weighted`.
- **estimator_permutation_invariance:** shuffling the estimator axis leaves the output unchanged (all except explicitly weighted ones given fixed `w`).
- **mean_bounds:** `mean` output lies within the per-class min/max across estimators.
- **median_robust_to_outlier:** 7 agreeing estimators + 1 adversarial one: `median` and `trimmed_mean` stay closer to the consensus than `mean`.
- **logit_mean_vs_mean_on_disagreement:** on a hand-built 2-estimator example, assert the known expected numbers (compute by hand in the test).
- **entropy_weighted_prefers_confident:** a near-one-hot estimator gets more weight than a near-uniform one at low `T`; at huge `T` it approaches `mean`.
- **weighted_matches_mean_for_uniform_w:** `weighted` with `w = 1/E` equals `mean`.
- **val_weighted_uses_only_validation_inputs:** pass poisoned test-side arrays; result must not change (guards against leakage).
- **mean_temp_offline_improves_or_matches_val_logloss:** fitted `T` gives validation log-loss <= `T=1` log-loss.
- **logit_mean_equals_softmax_of_mean_logits:** matches `softmax(mean_e L_e / T)` exactly on random logits.
- **temperature_applied_before_aggregation:** `mean` of `softmax(L/T)` differs from `softmax(mean L)` followed by temperature in a hand-built case, and equals the former.
- **regression_probs_valid:** `mixture` and `log_pool` return finite log-probabilities whose exponentials sum to 1 per row; decoded quantiles are non-decreasing in the level.
- **quantile_mean_known_value:** hand-checked small example on a histogram `BinDistribution`. `mixture` of two identical estimators equals either one.
- **registry_complete:** every name in the config's aggregator list exists in the registry; unknown name raises `KeyError` with a helpful message.

### 7.2 `test_coresets.py`
For each method in the registry (parametrized):
- **size_and_uniqueness:** `len(idx) == min(budget, n)`, indices unique, `0 <= idx < n`, integer dtype.
- **deterministic_given_seed / differs_across_seeds** (the latter not applied to fully deterministic cases like `budget >= n`).
- **budget_ge_n_returns_all:** result is a permutation of `range(n)`.
- **budget_zero_or_negative_raises.**
- **class_coverage:** for classification methods except `random`, every class appears when `budget >= n_classes`.
- **stratified_proportions:** class fractions in the subset are within 1/budget + 0.02 of the full-data fractions.
- **balanced_is_balanced_and_flagged:** per-class counts differ by at most 1 when classes are large enough; method metadata marks it as prior-shifting.
- **kmeans_picks_real_points:** every returned index is a valid row (no synthetic centroids returned).
- **uncertainty_mix_fraction:** with `frac_hard=0.5`, at least about half the selected points are in the top-uncertainty set of the pilot model (use a seeded toy where uncertainty is obvious).
- **no_label_leak_to_test:** call selection with test rows appended to the array and assert the selected indices never fall in the test range when the caller passes only training arrays (guards the calling convention in `experiments.py`).
- **perest_independent / perest_partition_cover:** `perest_random` gives subsets that differ across estimators; with `E*budget >= n`, the union of `perest_partition` subsets covers all rows and each subset has size `budget`.
- **handles_missing_class_downstream:** a coreset missing a class still yields aligned `(E, n, C)` probabilities via `align_proba` (no shape error, missing class prob ≈ 0).

### 7.3 `test_metrics.py`
- **log_loss_matches_sklearn** on random probabilities and labels (atol 1e-9); perfect predictions give ~0; uniform predictions give `log(C)`.
- **accuracy_known_cases.**
- **roc_auc_matches_sklearn** (binary and OvR multiclass); returns 0.5 for constant scores.
- **ece_bounds:** in `[0, 1]`; ~0 for perfectly calibrated synthetic data (large n, tolerance 0.02); high for a confidently-wrong predictor.
- **rmse_known_value.**
- **crps_from_quantiles:** zero when all quantiles equal the true value; increases when the distribution is shifted away; matches a hand-computed pinball average on a tiny example.
- **coverage:** central 80% interval of a correct Gaussian predictor covers about 80% (tolerance 0.03, seeded, n large).
- **metric_direction_registry:** every metric declares whether higher or lower is better; `stats.compare` uses it.

### 7.4 `test_stats.py`
- **bootstrap_ci_covers_true_effect:** simulate paired differences with known mean; CI contains it in at least ~90% of 200 repetitions (`slow`).
- **null_effect_rarely_significant:** differences with mean 0; "significant" flag fires in at most ~10% of repetitions at the 0.05 rule (`slow`).
- **sign_test_known_p:** e.g. 9 wins out of 10 gives the exact binomial p-value.
- **all_zero_differences_handled:** Wilcoxon wrapper returns p = 1.0, no exception.
- **win_rate_ties_half.**
- **dataset_weighting:** a dataset with more seeds must not outweigh others (compare to a manually averaged table).
- **direction_aware_compare:** for a lower-is-better metric, a smaller value is reported as an improvement.

### 7.5 `test_cache.py`
- **roundtrip_exact:** `put` then `get` returns identical arrays and dtypes.
- **key_sensitivity:** changing any one of dataset, seed, n_estimators, backend kwargs, or subset id changes the key; reordering dict kwargs does not.
- **hit_skips_backend:** with a counting fake backend, a second `run_aggregation` call makes zero backend calls.
- **atomic_write:** a simulated failure mid-write leaves no partial file that `exists()` would accept.
- **cached_equals_recomputed:** cached outputs equal a fresh run with the sklearn backend (same seed).

### 7.6 `test_protocol.py` (experimental-validity guards)
- **split_disjoint_and_complete:** train and test indices are disjoint and cover all rows; stratified for classification.
- **same_split_across_methods:** within one (dataset, seed), all methods are scored on the identical test rows (hash the test indices in the result metadata or recompute).
- **coreset_selected_from_train_only:** wrap `select_coreset` with a spy; assert it never receives an array longer than `n_train` and that returned indices map into train.
- **budget_not_above_n_train:** requested budgets above `n_train` are clipped and recorded as `full`, not silently duplicated.
- **seed_reproducibility:** the same config run twice (sklearn backend) gives identical CSVs.
- **different_seeds_differ:** different seeds give different splits and different metric values.
- **backend_called_with_correct_shapes:** the fake backend asserts shapes and `classes` ordering on every call.
- **test_labels_never_reach_backend:** the fake backend asserts `y_test` is never passed to any method.

### 7.7 `test_experiment_sanity.py` (checks that the experiments themselves behave; use smoke config)
- **null_experiment:** compare `random` vs `random` with different selection seeds across many cells; the stats layer must not call it significant (guards against false positives from analysis bugs) (`slow`).
- **canary_bad_baseline:** add a deliberately bad method (labels shuffled in the context); every real method must beat it on log-loss across datasets. If this fails, the pipeline is broken.
- **full_budget_equals_full:** a coreset method with `budget >= n_train` gives metrics identical to the `full` row.
- **more_data_helps_on_average:** mean log-loss at the largest budget is no worse than at the smallest budget by more than a small tolerance (averaged over datasets and seeds) (`slow`).
- **ensembling_helps_or_ties:** `mean` with 8 estimators is no worse than `single` on average by more than a small tolerance (`slow`).
- **aggregator_equals_mean_when_degenerate:** with `E=1`, all aggregators match `single` exactly.
- **no_nan_or_inf_in_results** for the full smoke grid.
- **regression_path_runs:** the regression pipeline (quantiles, `rmse`, `crps`) runs end to end on the smoke config.

### 7.8 `test_validate.py` (the results validator catches injected faults)
`validate_results(df, expected_methods, expected_datasets, expected_seeds, expected_budgets)` must report a problem for each of the following, tested by corrupting a good smoke-run DataFrame:
- missing column, wrong dtype, unknown metric name;
- NaN or inf value; duplicate (dataset, seed, method, budget, metric) rows;
- out-of-range values (`accuracy`/`roc_auc`/`ece` outside [0, 1], negative `log_loss`, negative `rmse`/`crps`);
- incomplete grid (a method, seed, dataset, or budget missing);
- budget larger than `n_train` for a non-`full` method.
A clean DataFrame returns an empty list.

### 7.9 `test_tabpfn_integration.py` (`@pytest.mark.tabpfn`, real model)
These run on the GPU server. They run by default when `tabpfn` is importable and CUDA is available (override with `RUN_TABPFN=0`); on machines without a GPU they skip.
- **shapes_and_normalisation:** `(E, n, C)`, finite, rows sum to 1.
- **estimators_actually_differ:** different `random_state` give non-identical outputs.
- **accuracy_above_trivial:** on a simple synthetic task, accuracy beats the majority-class baseline.
- **regression_outputs_valid:** `RegOutputs.probs` has shape `(E, n, B)`, finite, rows sum to 1; decoded quantiles are non-decreasing in the level and the median is close to the target on an easy task.
- **coreset_missing_class_ok:** training on a subset missing a class still returns aligned probabilities.
- **seed_reproducible:** same `random_state` gives the same output (GPU nondeterminism tolerance 1e-4).
- **baseline_reproduces_native_clf:** with one fitted classifier, `mean` applied to `predict_raw_logits` output (using `softmax_temperature_`) equals `predict_proba` (atol about 1e-4). `logit_mean` equals the probabilities of a classifier built with `average_before_softmax=True`. This replaces any "proxy" check: the offline aggregators are verified against the package itself.
- **baseline_reproduces_native_reg:** `mixture` run on the captured per-estimator distributions reproduces `reg.predict` for `mean`, `median` and quantiles (tolerance set after a first measurement); `log_pool` matches `average_before_softmax=True`. This test guards the private-API capture loop, so it must pass before any regression result is trusted.
- **class_order_aligned:** with non-contiguous or string labels, the logit columns match `clf.classes_` and `argmax` of the mean probabilities equals `predict`.
- **random_state_controls_estimators:** different `random_state` give different per-estimator outputs and different `n_estimators` members; the same `random_state` repeats (within GPU tolerance). Guards the "default seed is 0" trap.
- **n_estimators_recorded:** the requested E equals `n_estimators_` on narrow data; on very wide data any auto-scaling is surfaced in the metadata.
- **resolved_config_saved:** `meta` contains the tabpfn version, the model version, and the resolved inference config dict.
- **builtin_subsample_baseline_runs:** `inference_config={"SUBSAMPLE_SAMPLES": k}` runs, differs from full-context predictions, and the allowed `SampleSubsamplingMethod` values are read from the package (no hard-coded strings).
- **embedding_shapes:** `get_embeddings` returns `(E, n, dim)`, and `data_source="train"` raises a clear error under `fit_with_cache`.
- **batched_matches_unbatched (if enabled):** `predict_proba_batched` equals per-dataset `predict_proba` within tolerance on same-shape datasets.
- **oom_is_recorded_not_fatal:** force a failure (e.g. monkeypatch the backend to raise `torch.cuda.OutOfMemoryError`) and assert the cell lands in the failures CSV, the sweep continues, and `validate_results` flags the missing cell.
- **shard_union_equals_full_grid:** running shards `0/3, 1/3, 2/3` on the smoke config and merging gives exactly the same rows as an unsharded run (this one uses the sklearn backend and does not need a GPU).
- **resume_skips_finished_cells:** kill a run halfway (raise after k cells), rerun, and assert finished cells were not recomputed and the final CSV is complete with no duplicates.

### 7.10 `test_sizes.py` (behaviour across dataset sizes)
Parametrize over `n in {300, 3_000, 30_000}`, `ratio in {0.5, 0.1, 0.01}` and every coreset method unless noted.
- **contract_across_sizes_and_ratios:** the section 3.3 contract holds, including the edge cases `budget = n - 1`, `n`, `n + 1`, and `budget = n_classes` and `budget = 1` (the latter either works or raises a clear error, never returns the wrong size).
- **extreme_reduction:** at `ratio = 0.001` (clipped up to a sane minimum) every method still returns valid indices and every class when `budget >= n_classes`.
- **selection_time_scales:** selection time at `10 n` is under about 30x the time at `n` (linear-ish, generous constant) (`slow`). Methods with expensive internals (`kmeans_stratified`, `uncertainty_mix`, `embedding_kmeans`) must bound their cost through `budget` or `pilot` caps, not `n`.
- **selection_memory_bounded:** peak memory during selection (use `tracemalloc` on the numpy path) stays under about 10x `X.nbytes` at `n = 30_000`. This catches accidental `n x n` distance matrices that would explode at 1M rows.
- **imbalanced_classes:** with 1% minority class at each size, `stratified` and `kmeans_stratified` keep at least one minority row when `budget >= n_classes`; `random` is allowed to miss and the test records how often.
- **many_classes:** 10 and 50 classes at each size: no empty classes in the context for the class-aware methods.
- **wide_tables:** 5, 50 and 500 features at `n = 3_000`: distance-based methods run (they must reduce dimension above a cap, e.g. PCA to at most 50 dims) and return the same-size output.
- **messy_inputs:** NaNs, constant columns, duplicate rows, categorical codes, and heavy-tailed regression targets all yield valid, deterministic selections.
- **regression_tail_coverage:** with a heavy-tailed target, target-bin stratification includes at least half of the proportional share of the top 1% of targets.
- **resolve_budgets_rules:** from fractions and absolutes: unique, sorted, `>= n_classes`, budgets `>= n_train` dropped, clipping logged; identical inputs give identical outputs.
- **tier_assignment:** `tier_of` is consistent at the boundaries (999, 1000, 99,999, 100,000).
- **runner_handles_mixed_tiers:** a smoke run with datasets from three tiers produces a `tier` column, one row per (dataset, method, budget, seed, metric), no silently dropped dataset (the validator checks), and per-tier tables weight every dataset equally.
- **cache_keys_include_size_factors:** budget, method, `n_train` and tier-relevant settings all change the cache key.
- **tier_large_smoke (`tier_large`, GPU):** one real large dataset (`n_train >= 100k`) at ratios 0.01 and 0.1 with the real backend completes, and `full` completes on the same data (or the cell lands in the failures CSV, not a crash).

### 7.11 `test_effect_sizes.py` (the numbers that decide experiment 2)
Use `make_synthetic` learning curves with known parameters so every expectation has ground truth.
- **equivalent_budget_recovers_known_multiplier:** simulate a "coreset" whose metric at budget `b` equals the random metric at `k * b` for known `k` in `{1, 2, 5}`, add noise, and recover `k` within 15%.
- **equivalent_budget_no_silent_extrapolation:** if the coreset beats the best random budget measured, or is worse than the smallest, return `NaN` plus a flag (`above_range` / `below_range`), never an extrapolated number.
- **learning_curve_is_smoothed_monotone:** a noisy, non-monotone random curve is made monotone (isotonic regression) before inversion, and the result is stable across noise seeds.
- **gap_closed_formula:** hand-computed cases for lower-is-better and higher-is-better metrics; 0 at random, 1 at full, may exceed 1 or go below 0 and is not clipped.
- **gap_closed_degenerate_headroom:** when `|random - full|` is below the noise floor, the result is `NaN` and the cell is flagged `saturated`, never `inf` or a huge ratio.
- **saturated_cells_excluded_from_claims:** `compare` and the report skip flagged cells in significance tests but list them.
- **paired_selection_beats_unpaired:** with a fixed test split and selection seed as the only varying factor, the variance of paired differences versus `random` is lower than when the split also varies (verifies the protocol choice on simulated data).
- **bootstrap_resamples_datasets:** simulate `D` datasets with `S` seeds each; as `S` grows the CI width stops shrinking at the dataset-level floor, and shrinks like `1/sqrt(D)` as `D` grows.
- **type_I_error_controlled:** with zero true effect, over 1,000 simulated experiments the rate of "significant" results is at most about 0.07 at alpha 0.05, both uncorrected for a single comparison and Holm-corrected over a family of (method x budget) comparisons (`slow`).
- **power_meets_design:** for the default design (`D` datasets, `S` seeds) and the declared `min_detectable_effect`, simulated power is at least 0.8 (`slow`).
- **min_detectable_effect_monotone:** the planner's minimum detectable effect decreases as `D` or `S` grows and increases with noise; tier noise (small tiers noisier) is respected.
- **holm_correction:** known p-values give known adjusted values; adjusted p is never below raw p, is monotone in sorted order, and caps at 1.
- **heterogeneity_by_tier:** if the simulated effect exists only in the small tier, the per-tier analysis detects it there and not in the large tier, and the pooled analysis reports the heterogeneity instead of averaging it away.
- **ratio_curve_summary:** the `gap_closed` versus reduction-ratio table has one row per ratio, sorted, and handles missing cells without error.

### 7.12 `test_design.py` (experimental discipline)
- **dev_confirm_disjoint:** the config validator rejects overlapping `dev` and `confirm` datasets.
- **tuning_only_on_dev:** hyperparameter-tuning helpers raise if handed a `confirm` dataset; a test spies on dataset names passed to tuning code.
- **family_defined_before_running:** the list of (method, budget, metric) comparisons used for Holm is written to the results metadata at launch, and `analyze` uses that list, not whatever was convenient afterwards.
- **plan_blocks_underpowered_runs:** `cli plan` exits non-zero for a design whose power at `min_detectable_effect` is below 0.8 and succeeds with `--force`.
- **baselines_present_at_every_cell:** every (dataset, budget, seed) contains `full`, `random`, `builtin_subsample` and `builtin_majority_downsample` rows, otherwise the validator fails the run (a coreset can't be judged without them).
- **random_learning_curve_present:** at least 6 log-spaced `random` budgets exist per dataset for the equivalence calculation.

### 7.13 `test_estimator.py` (benchmark-ready wrapper)
- **sklearn_contract:** `clone`, `get_params`/`set_params`, `check_is_fitted`, and `fit`/`predict` on pandas and numpy input work for both classifier and regressor (sklearn backend).
- **reproduces_stock_when_default:** with `aggregator="mean"`, `coreset="none"` the wrapper's probabilities equal the stock estimator's (`tabpfn` backend, `@pytest.mark.tabpfn`); with the sklearn backend, equal to the backend's own mean path.
- **coreset_selected_in_fit_only:** a spy shows selection sees only the arrays passed to `fit`, never `predict` inputs.
- **validation_slice_from_train_only:** for `val_weighted`/`mean_temp_offline`, poisoned `predict` inputs do not change fitted weights.
- **string_and_noncontiguous_labels:** predictions map back to the original labels and class order.
- **budget_fraction_resolved_at_fit:** `budget_fraction=0.1` resolves against the `fit` size; budget at or above `n_train` means no reduction.
- **deterministic_given_random_state, picklable, clonable inside a cross-validation loop** (`sklearn.model_selection.cross_val_score` with a tiny dataset).
- **no_benchmark_code_in_core:** importing `tabpfn_lab` must not import any benchmark package.

---

## 8. Build order for Claude Code

1. `pyproject.toml`, package skeleton, `CLAUDE.md`, `.gitignore` (ignore `results/`).
2. `metrics.py` + `test_metrics.py`.
3. `backends.py` (`align_proba`, `SklearnBackend`) + `datasets.py` (synthetic loaders, `Split`, `make_split`) + `test_protocol.py` split tests.
4. `aggregation.py` + `test_aggregation.py`.
5. `coresets.py` + `test_coresets.py`.
6. `cache.py` + `test_cache.py`.
7. `stats.py` + `test_stats.py`, `validate.py` + `test_validate.py`.
8. `experiments.py` + `cli.py` + `smoke.yaml`; then `test_experiment_sanity.py` and remaining `test_protocol.py` tests.
8b. `sizes.py` + `test_sizes.py`, then `effect.py` + `test_effect_sizes.py`, then the plan/Holm/dev-confirm pieces + `test_design.py`. Do these before the real GPU runs, because they define how results will be judged.
9. `TabPFNBackend` (including `native_clf`/`native_reg` and the regression capture loop) + OpenML loaders + `test_tabpfn_integration.py`; add sharding, failure logging and the merge command to the CLI.
10. `estimator.py` + `test_estimator.py` (benchmark-ready wrapper), then a `benchmarks/` folder with one small adapter per benchmark you want (read each benchmark's docs for its registration mechanism).
11. README with the exact commands: `pip install -e ".[dev,tabpfn]"`, `pytest` (fast, sklearn backend), `pytest -m tabpfn` (real model, on the GPU box), `python -m tabpfn_lab.cli aggregation --config configs/smoke.yaml`, and a server example:
    `CUDA_VISIBLE_DEVICES=0 nohup python -m tabpfn_lab.cli coreset --config configs/coreset.yaml --shard 0/2 > results/logs/c0.out 2>&1 &`
    (and the same with GPU 1 and shard `1/2`), then `python -m tabpfn_lab.cli merge --name coreset_v1`.

After each step, run `pytest` and keep it green before moving on.

---

## 9. Open items to verify (do not guess)

- **Resolved from source (no longer open):** checkpoint selection is `ModelVersion` plus `create_default_for_version`; the default aggregation is the mean of per-estimator softmax probabilities after temperature (classification) and a probability mixture (regression); per-estimator classification logits are available via `predict_raw_logits`.
- Still to read from the installed source: `tabpfn/preprocessing/` (the members of `SampleSubsamplingMethod`, and what `TabPFNEnsemblePreprocessor` exposes per estimator, e.g. `subsample_row_indices`); `tabpfn/downsample_correction.py` (exact semantics of `downsample_class_weights`); `tabpfn/utils.py` (whether `translate_probs_across_borders` expects logits or probabilities, which the capture loop must match); `tabpfn/errors.py` (the OOM exception class raised by `handle_oom_errors`, to catch alongside `torch.cuda.OutOfMemoryError`).
- The resolved inference config for the 3.5 checkpoint: dump it once to see which preprocessing transforms and target transforms are really active, and what the checkpoint's `N_ESTIMATORS` and `SOFTMAX_TEMPERATURE` are.
- Optional third experiment: a mixed-checkpoint ensemble (`model_path=[v3, v3.5]`, which the package says applies the models across estimators). Check that mixing the v3 and 3.5 checkpoints is accepted for both tasks before building on it.
- The open 3.5 weights are under a non-commercial licence: keep this repo research-only.
- Whether downloading the 3.5 weights on the server needs a Hugging Face login or licence acceptance step; if so, document it in the README and make the backend fail with a clear message.
- Real-data runs should use datasets with `n_train` well above the largest budget; pick them from TabArena-medium, the BeyondArena large subset, or the TALENT large extension.
