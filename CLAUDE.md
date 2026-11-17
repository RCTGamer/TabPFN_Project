# tabpfn-lab conventions

- Spec: `PROJECT_SPEC.md` is the source of truth; build order is its section 8.
- **Never import `tabpfn` (or `torch`) outside `src/tabpfn_lab/backends.py`.** Experiments talk to the `Backend` interface only.
- Anything that selects rows or fits weights sees training labels only: test features may be used at prediction time
  (routing, gating, `local_mix`), test labels never. Methods whose prediction for a row depends on other test rows
  carry `flags = "transductive"`. Tests enforce this.
- Everything is seeded; same config + seed must give bit-identical CSVs with the sklearn backend.
- Results are long-format with the schema in `validate.py`; run `validate_results` on anything you produce.
- New aggregator = one function + one line in `CLF_AGGREGATORS`/`REG_AGGREGATORS`. New coreset = one function + one line in `CORESET_METHODS`.

## Tests
- `pytest` — fast, sklearn backend only (tabpfn tests skip without tabpfn+CUDA).
- `pytest -m "not slow"` — skip simulation-heavy tests.
- `pytest -m tabpfn` — real model, on the GPU server.
- Keep `pytest` green after every change; run `python -m tabpfn_lab.cli aggregation --config configs/smoke.yaml` before long GPU runs.

## Results log

Facts from finished runs, not instructions. Append each new run: config, runs per dataset, main table, findings.
All runs so far: TabPFN 3.5 on one RTX A6000, **dev datasets only** (exploratory; the methods were designed after
looking at these datasets, so nothing here is a confirmed claim). 8 estimators, 1024 rows per context, baseline
`perest_random`. Lower is better: log loss for classification, RMSE for regression; values are means over runs.
Not recorded yet: the aggregation runs and `perest_shared`. Runs A-D below are from before runs 1-3; A is not an
8 x 1024 run. Results live on the A6000 server under `results/` (git-ignored): `tables/<name>.csv`,
`<name>.timing.csv`, `<name>.meta.json`, `<name>.failures.csv`, `<name>.analysis.*.csv`, and `logs/<name>.log`.

### A. `scaling_dev.yaml` as `scaling_a6000` (2026-10-28, 5 runs per dataset): context size x estimator count

Random nested contexts 512, 1024, 2048, ... up to all training rows, crossed with E = 1, 2, 4, 8; each cell its own
timed fit. Datasets (70/30 split): phoneme 3,782 train rows x 5 features, houses 14,448 x 8, electricity 31,718 x 8,
higgs 68,635 x 28.

| dataset | 512 rows, E=1 | full, E=1 | full, E=8 | seconds full E=1 / E=8 |
|---|---|---|---|---|
| electricity | 0.431 | 0.081 | 0.076 | 3.5 / 18.4 |
| higgs | 0.593 | 0.502 | 0.501 | 10.3 / 71.6 |
| houses | 51,133 | 37,559 | 37,187 | 2.1 / 8.0 |
| phoneme | 0.330 | 0.181 | 0.178 | 1.6 / 2.5 |

- More context beats more estimators at equal time (electricity: E=1 at 16,384 rows 0.125 in 2.3 s vs E=8 at 8,192
  rows 0.176 in 5.7 s). E beyond 2 adds little; on electricity E=2 was often better than E=8.
- electricity still improves about 30% per doubling at full size; higgs about 0.01 per doubling.
- `fit_seconds` is a near-constant 1.6 to 2 s per fit regardless of context size (likely model loading); time grows
  faster than linearly with context at large sizes (higgs E=8: 29 s at 32,768 rows, 72 s at 65,536).
- No out-of-memory on the A6000, including higgs full with E=8.

### B. `perest_dev.yaml` (2026-10-29, 9 runs per dataset): per-estimator contexts

| method | electricity | houses | phoneme | seconds (elec / houses / phon) |
|---|---|---|---|---|
| perest_random | 0.365 | 44,337 | 0.251 | 4.6 / 22.2 / 3.3 |
| perest_partition | 0.365 | 44,079 | 0.251 | 4.7 / 22.3 / 3.4 |
| builtin_subsample | 0.366 | 44,250 | 0.249 | 3.6 / 21.1 / 2.3 |
| perest_stratified | 0.366 | 44,356 | 0.253 | 4.7 / 21.8 / 3.2 |
| perest_representative | 0.368 | 44,209 | 0.253 | 4.7 / 21.8 / 3.2 |
| perest_kcenter | 0.458 | 49,571 | 0.323 | 9.9 / 22.3 / 4.0 |
| cluster_experts | 0.457 | 54,658 | 0.353 | 5.9 / 38.6 / 3.7 |
| cluster_experts+gate | 0.260 | 43,264 | 0.184 | (same cell) |
| local_mix_r0.5 | 0.282 | 43,305 | 0.211 | 18.2 / 36.1 / 16.8 |
| local_mix_r0.75 | 0.263 | 43,378 | 0.212 | 18.5 / 35.4 / 16.4 |
| local_mix_r0.9 | 0.258 | 43,695 | 0.215 | 17.8 / 36.3 / 16.2 |
| local_mix_r1 | 0.263 | 44,363 | 0.217 | 17.9 / 35.9 / 16.7 |
| full | 0.077 | 37,427 | 0.178 | 20.8 / 25.6 / 2.7 |

- All random-like variants (partition, builtin, stratified, representative) equal perest_random.
- kcenter hurts everywhere: farthest-point covers pick outliers and distort the row distribution.
- local_mix helps on all three; some random share is safer (houses: r0.5 gains, r1 none).
- cluster_experts+gate is best or equal-best and costs one fit; local_mix costs 8 fits (one per test group, about
  2 s fixed each), which dominates its time on phoneme.
- houses `cluster_experts` is slow only because its cell scores two regression rows (plain and gate); see the
  scoring note under "Harness checks".

### C. `perest_crumb.yaml` (2026-10-30, 9 runs per dataset): CRUMB recreation

perest_random and full match run B exactly (same splits and seeds), so B and C are directly comparable.

| method | electricity | houses | phoneme | seconds (elec / houses / phon) | gap closed (elec / houses / phon) |
|---|---|---|---|---|---|
| crumb | 0.234 | 43,584 | 0.177 | 25.3 / 43.0 / 22.6 | 45% / 11% / 101% |
| crumb_perest | 0.200 | 41,816 | 0.196 | 36.2 / 48.9 / 23.7 | 57% / 36% / 75% |
| (B) cluster_experts+gate | 0.260 | 43,264 | 0.184 | 5.9 / 38.6 / 3.7 | 37% / 16% / 91% |

- CRUMB beats local_mix on all three; crumb_perest is best on electricity and houses, crumb matches full on phoneme
  (with 3.8k train rows, crumb_perest's 70% candidate subsample removes useful rows).
- Costliest method: 8 fits plus herding (about 30 to 40 s of CPU per electricity cell). Both CRUMB rows are
  transductive.
- On these datasets `full` beats every 8 x 1024 method in quality and time (phoneme: equal quality, 8x faster).

### D. `scripts/resample_test.py` (2026-11, phoneme, 5 seeds, E = 1 and 8): duplicated or bootstrapped context

Training split inflated to 4x by exact duplication (`dup4`) or bootstrap (`boot4`), plus a same-size bootstrap
control (`boot1`), against the plain training set. Result as reported: no improvement over `full`; inflating the
context only added time. Numbers not logged here (rerun the script to print the table).

### 1. `perest_gate.yaml` (2026-11-16, 9 runs per dataset): gate temperature fitted on held-out training rows

| dataset | perest_random | cluster_experts | +gate (fixed) | _tuned | _tuned+heuristic | full |
|---|---|---|---|---|---|---|
| electricity | 0.365 | 0.457 | 0.260 | 0.231 | 0.257 | 0.077 |
| houses | 44,337 | 54,658 | 43,264 | 43,073 | 43,085 | 37,427 |
| phoneme | 0.251 | 0.353 | 0.184 | 0.200 | 0.200 | 0.178 |

- Without a gate, cluster experts lose to random contexts everywhere; with the fixed gate they win on all three.
- Tuning alone (`_tuned` vs `_tuned+heuristic`, identical experts) helped only electricity (0.257 to 0.231).
- phoneme lost 8% against the fixed gate because the 20% held-out slice shrinks a small training set.
- Fitted temperature, as a multiple of the median heuristic: 0.04 to 0.32 on electricity, 0.6 to 1.0 on houses,
  scattered on phoneme. The per-expert offset (`+bias`) added nothing.

### 2. `perest_gate_large.yaml` (2026-11-17, 6 runs per dataset): the same methods on six larger datasets

| dataset | perest_random | cluster_experts | +gate (fixed) | _tuned | _tuned+heuristic | full |
|---|---|---|---|---|---|---|
| electricity | 0.365 | 0.453 | 0.259 | 0.229 | 0.256 | 0.076 |
| jungle_chess | 0.331 | 0.642 | 0.413 | 0.263 | 0.417 | 0.113 |
| jannis | 0.719 | 0.867 | 0.790 | 0.707 | 0.784 | 0.577 |
| higgs | 0.552 | 0.578 | 0.562 | 0.553 | 0.562 | 0.500 |
| house_16H | 30,730 | 34,476 | 29,829 | 29,667 | 29,775 | 25,226 |
| fried | 1.005 | 1.085 | 1.038 | 1.015 | 1.026 | 0.999 |

- The fixed gate loses to random contexts on 4 of 6: it is not robust on large data.
- The tuned gate beats the fixed one on all 6; the fitted temperature was below the heuristic in 35 of 36 fits.
- `nearest_is_best` (share of held-out rows whose nearest expert is the best one) tracks the gain: about 0.6
  electricity, 0.52 jungle_chess, 0.35 jannis, 0.23 house_16H, 0.21 higgs, 0.11 fried (chance is 0.125).
- fried sits at its noise floor (RMSE about 1.0): full context gains only 0.6%, so no context method can gain.
- `full` fitted on all six without running out of GPU memory.

### 3. `perest_experts_large.yaml` (2026-11-17, 3 runs per dataset): more clusters, feature weighting, overlap

| dataset | perest_random | cluster_experts_tuned | experts_scaled | _weighted | _overlap | experts_full | full+hard | full+heuristic | full | gap closed |
|---|---|---|---|---|---|---|---|---|---|---|
| electricity | 0.369 | 0.228 | 0.128 | 0.127 | 0.109 | 0.107 | 0.109 | 0.139 | 0.077 | 90% |
| jungle_chess | 0.331 | 0.260 | 0.167 | 0.159 | 0.149 | 0.149 | 0.151 | 0.187 | 0.113 | 83% |
| jannis | 0.718 | 0.707 | 0.680 | 0.668 | 0.682 | 0.664 | 0.676 | 0.687 | 0.577 | 39% |
| higgs | 0.551 | 0.553 | 0.544 | 0.542 | 0.542 | 0.540 | 0.548 | 0.544 | 0.500 | 23% |
| house_16H | 30,735 | 29,439 | 29,157 | 28,721 | 28,435 | 28,052 | 28,273 | 28,669 | 25,237 | 49% |
| fried | 1.005 | 1.018 | 1.023 | 1.025 | 1.023 | 1.018 | 1.035 | 1.020 | 0.999 | none |

"Gap closed" = share of the distance from `perest_random` to `full` recovered by `experts_full`.

- Clusters (without / with overlap): electricity 30/59, jungle_chess 29/58, jannis 56/111, higgs 66/128 (the
  `max_clusters` cap), house_16H 14/28, fried 26/52.
- More clusters is the biggest gain (electricity 0.228 to 0.128). Overlap adds 11 to 15% on electricity and
  jungle_chess; weighting adds 1.5 to 5% on jungle_chess, jannis and house_16H. Both together are best wherever any
  variant beats random. Weighting did not raise `nearest_is_best` on higgs (about 0.20).
- The fitted soft gate beats nearest-expert-only (`+hard`) on all six by 1 to 2%, and the untuned gate by up to 23%.
  `+hard` uses one ensemble member, so it is weaker than real MICP (all 8 estimators on the nearest context).
- Cost: `total_seconds` of `experts_full` vs `full`: electricity 107 vs 24, jungle_chess 110 vs 17, jannis 220 vs 61,
  higgs 251 vs 70, house_16H 115 vs 29, fried 217 vs 57. At these sizes full context is better and 3.5 to 6x cheaper.
  Each context is currently processed in about 8 fits (once per neighbouring centre).

### Harness checks and known issues

- On the GPU server the real-model tests pass on an RTX A6000.
- Resume counts a cell as done if any of its rows exist, so a run killed mid-write can drop rows, and
  `validate_results` does not check that every cell has all its metrics. A half-written last line makes the CSV
  unreadable once resume appends to it.
- `plan` can never pass for `coreset.yaml`: with 5 confirm datasets the smallest two-sided sign-test p is 0.0625.
  A confirm set needs at least 6 datasets per task.
- `total_seconds` includes about 21 s of scoring per result row on regression; use `fit_seconds + predict_seconds`
  for model cost.

### Related work (from the papers, not our runs)

- CRUMB (Heredge et al., arXiv 2606.11473): standardised raw features, k-means on the test rows (K = 20), greedy MMD
  herding with an RBF kernel and median-heuristic bandwidth, context n = 0.1 N. Our `crumb` matches except K = 8,
  a fixed 1024-row budget and exact herding. On TabArena it beat MICP, which beat uniform subsampling.
- MICP (MixturePFN): k-means on the training rows into N / n clusters, one support set per cluster, hard routing to
  the nearest centre. This is `cluster_experts` with hard routing and a scaled cluster count.
- QCOC (arXiv 2610.11784): retrieval in a learned embedding space did no better than standardised raw features;
  most of CRUMB's gain came from grouping the queries rather than from the herding.

## Viewing a run (on the server, repo root, `.venv` active)
- Status: `pgrep -u $USER -af tabpfn_lab.cli`; `tail -2 results/logs/<name>.log`;
  `python -m tabpfn_lab.cli validate --csv results/tables/<name>.csv`.
- Tables: `python -m tabpfn_lab.cli analyze --csv results/tables/<name>.csv`. Scaling writes `.analysis.summary.csv`,
  `.analysis.context_steps.csv`, `.analysis.estimator_steps.csv`; perest runs write `.analysis.dev.csv` (paired vs
  the config's baseline, Holm-adjusted, with mean seconds).
- Compare perest runs by concatenating their `.analysis.dev.csv` files (same splits and seeds give identical
  perest_random and full rows). Cost: group `<name>.timing.csv` by dataset and method over fit, predict, selection
  and total seconds and n_fits.
