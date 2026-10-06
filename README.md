# TabPFN Project

A place to learn how TabPFN behaves, then test whether custom preprocessing
makes inference faster or reduces GPU memory, without losing accuracy.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The default TabPFN weights are **gated on Hugging Face**. Accept the license at
<https://huggingface.co/Prior-Labs/tabpfn_3_5>, then run `hf auth login` or
export `HF_TOKEN=<read token>`. Without the weights, the tests that need the
model are skipped and everything else still runs.

```bash
pytest                    # all tests (model tests skip if weights are missing)
pytest -s -m model        # only the TabPFN comparisons, with timing output
pytest -m "not model"     # fast unit tests only
python scripts/compare_preprocessing.py --openml credit-g --n-estimators 4
```

## Layout

```
src/tabpfn_lab/
  preprocessing.py   # your custom steps (column transformers + row reducers)
  benchmark.py       # run_benchmark(): fit/predict time, peak CUDA memory, accuracy
tests/
  test_preprocessing.py         # fast, no model: each step does what it claims
  test_benchmark.py             # the measurement harness itself (uses LogisticRegression)
  test_tabpfn_preprocessing.py  # baseline TabPFN vs. TabPFN + preprocessing
scripts/
  compare_preprocessing.py      # many configs x repeats -> results/compare.csv
```

## Outline

### Phase 1: Learn how TabPFN works
1. Fit `TabPFNClassifier` / `TabPFNRegressor` on small sklearn datasets
   (breast cancer, iris, diabetes) and compare against a GBM baseline.
2. Understand the cost model. TabPFN does no gradient training: "fit" mostly
   preprocesses the data and stores it as the context. At predict time the
   transformer attends over **train rows + test rows** and over
   **features**, so time and memory grow with `n_train`, `n_features` and
   `n_estimators` (ensemble members, each with different internal preprocessing).
3. Learn the built-in settings before writing your own:
   - `n_estimators`: roughly linear cost
   - `fit_mode`: `low_memory` / `fit_preprocessors` (default) /
     `fit_with_cache` (KV cache of the train set, which makes repeated predicts faster but uses more memory)
   - `inference_precision`, `memory_saving_mode`, `kv_cache_precision`
   - `ignore_pretraining_limits` (row and feature limits)
4. Profile one fit/predict with `torch.profiler` or `torch.cuda.max_memory_allocated()`
   to see where the time actually goes. Check this before optimizing anything.

### Phase 2: Measurement harness (done: `benchmark.py`)
- Time `fit` and `predict` separately, with `torch.cuda.synchronize()`
  around each so asynchronous kernels are counted.
- Reset and read `torch.cuda.max_memory_allocated()` per phase.
- Treat repeat 0 as warm-up (CUDA context and weight loading), and report the
  median of the later repeats.

### Phase 3: Custom preprocessing experiments
There are two kinds of steps, and they plug in differently:

| Kind | Examples | How it plugs in |
|---|---|---|
| Column transforms (same on train & test) | drop constant cols, drop correlated cols, float32 cast, PCA / `SelectKBest` | sklearn `Pipeline` in front of TabPFN |
| Row reducers (train only) | deduplicate, stratified subsample, k-means prototypes, coreset selection | `row_reducers=[...]` in `run_benchmark`, applied before `fit` |

Row reducers are where the large GPU savings are likely to be, because the
train context is the expensive part. Ideas to try next:
- Cluster-based prototypes (k-means per class) instead of random subsampling
- Pre-casting to float16 or pinned-memory tensors to reduce host-to-device transfer
- Caching the preprocessed train context once (`fit_with_cache`) and reusing it across many predict batches

### Phase 4: Evaluate
- Use the same configs on 3–5 OpenML datasets of different shapes (wide, tall, many classes).
- Track Δaccuracy (or ROC-AUC), Δfit time, Δpredict time, and Δpeak GPU MB, each relative to baseline.
- A step is a win only if it is faster or leaner and stays within the accuracy tolerance.

## The unit-test contract

`tests/test_tabpfn_preprocessing.py` encodes the rule for every new step:

1. The step must actually shrink what TabPFN sees (fewer rows or features).
2. Accuracy must stay within `ACCURACY_TOLERANCE` (0.03) of baseline.
3. On CUDA, peak fit memory must not go up.

Timings are printed but not asserted, because they are too noisy across machines
to use as a pass/fail check. Use the script for timing comparisons.

To test a new step: add it to `preprocessing.py`, write a small no-model test
in `test_preprocessing.py`, then add it to a pipeline or `row_reducers` list
in `test_tabpfn_preprocessing.py`.
