# Benchmark adapters

External benchmarks (TabArena, BeyondArena, TALENT, ...) register custom models in their own ways. Keep each
adapter here as a small module wrapping `tabpfn_lab.estimator.TabPFNLabClassifier` / `TabPFNLabRegressor`.
Never put benchmark code inside `src/tabpfn_lab`; `tests/test_estimator.py::test_no_benchmark_code_in_core`
checks that importing the package pulls in no benchmark package.

Rules that keep later claims valid (spec section 3.8):

- Run the **unmodified TabPFN-3.5** (`aggregator="mean"`, `coreset="none"`) in the same harness, on the same
  hardware and splits, as the baseline. Never compare against numbers copied from a report.
- Benchmark datasets belong to the `confirm` set: do not tune coreset or aggregator settings on them.
- The open weights are non-commercial: no commercial decisions or procurement comparisons from these results.

No adapters exist yet. Read each benchmark's documentation for its registration mechanism when you add one.
