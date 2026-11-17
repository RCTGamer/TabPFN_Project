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
Not recorded here yet: the `perest_dev`, `perest_crumb`, `scaling_*` and aggregation runs.

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
