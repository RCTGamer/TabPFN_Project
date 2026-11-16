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

## Experiments run so far (dev datasets, TabPFN 3.5, RTX A6000 server)
Results live on that server under `results/` (git-ignored, not in this repo):
- `results/tables/<name>.csv` metrics (long format), `<name>.timing.csv` fit / predict / selection seconds and peak GPU MB,
  `<name>.meta.json` config and grid, `<name>.failures.csv` failed cells, `<name>.analysis.*.csv` written by `analyze`.
- `results/logs/<name>.log` one line per cell.

Datasets (70/30 split): phoneme 5,404 rows x 5 features (n_train 3,782), houses 20,640 x 8 (14,448, regression),
electricity 45,312 x 8 (31,718), higgs 98,050 x 28 (68,635, scaling only).

| name | config | what | main finding |
|---|---|---|---|
| `scaling_a6000` | `configs/scaling_dev.yaml`, renamed copy | random nested contexts 512, 1024, ..., full x E = 1, 2, 4, 8; 4 datasets; 5 seeds | more context beats more estimators at equal time; E > 2 adds little; ~1.6-2 s fixed cost per fit |
| `perest_dev` | `configs/perest_dev.yaml` | E = 8, 1024 rows per estimator: perest_random / partition / stratified, builtin_subsample, perest_representative, perest_kcenter, local_mix_r{0.5,0.75,0.9,1}, cluster_experts (+gate); 3 datasets; 3 x 3 seeds | random-like variants all equal; kcenter and plain cluster_experts worse; local_mix and cluster_experts+gate clearly better; +gate best value per second and not transductive |
| `perest_crumb` | `configs/perest_crumb.yaml` | crumb, crumb_perest vs perest_random; same splits and seeds as perest_dev | CRUMB beats local_mix everywhere; crumb_perest best on electricity and houses, crumb matches full on phoneme; costliest (8 fits + herding) |

On these datasets `full` beats every 8 x 1024 method in quality and time: selection only pays off where the full
context does not fit. Pending: `perest_shared` (one shared random 1024 for all 8 estimators vs perest_random; config
created on the server only) and `scripts/resample_test.py` (phoneme duplicated / bootstrapped to 4x vs full).

How to view (on the server, repo root, `.venv` active):
- Status: `pgrep -u $USER -af tabpfn_lab.cli`, `tail -2 results/logs/<name>.log`,
  `python -m tabpfn_lab.cli validate --csv results/tables/<name>.csv`.
- Tables: `python -m tabpfn_lab.cli analyze --csv results/tables/<name>.csv`. Scaling writes `.analysis.summary.csv`,
  `.analysis.context_steps.csv`, `.analysis.estimator_steps.csv`; perest runs write `.analysis.dev.csv`, paired vs
  perest_random with Holm-adjusted p and mean seconds.
- Scaling grid: pivot `scaling_a6000.analysis.summary.csv` (index budget, columns n_estimators, values mean or total_seconds).
- Compare perest runs: concatenate `perest_dev.analysis.dev.csv` with the crumb rows of `perest_crumb.analysis.dev.csv`;
  perest_random and full are identical in both, which confirms they are comparable.
- Cost: group `<name>.timing.csv` by dataset and method over fit_seconds, predict_seconds, selection_seconds,
  total_seconds and n_fits.