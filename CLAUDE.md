# tabpfn-lab conventions

- Spec: `PROJECT_SPEC.md` is the source of truth; build order is its section 8.
- **Never import `tabpfn` (or `torch`) outside `src/tabpfn_lab/backends.py`.** Experiments talk to the `Backend` interface only.
- Anything that selects rows or fits weights sees the training split only. Tests enforce this.
- Everything is seeded; same config + seed must give bit-identical CSVs with the sklearn backend.
- Results are long-format with the schema in `validate.py`; run `validate_results` on anything you produce.
- New aggregator = one function + one line in `CLF_AGGREGATORS`/`REG_AGGREGATORS`. New coreset = one function + one line in `CORESET_METHODS`.

## Tests
- `pytest` — fast, sklearn backend only (tabpfn tests skip without tabpfn+CUDA).
- `pytest -m "not slow"` — skip simulation-heavy tests.
- `pytest -m tabpfn` — real model, on the GPU server.
- Keep `pytest` green after every change; run `python -m tabpfn_lab.cli aggregation --config configs/smoke.yaml` before long GPU runs.
