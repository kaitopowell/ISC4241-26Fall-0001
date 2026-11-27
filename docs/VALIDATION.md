# Validation checks

- `tests/test_pipeline.py` (8 tests): 11 complete seasons; changing a match's own score or shots leaves its features unchanged; changing all same-day and later matches leaves a match's features unchanged; odds never appear in any model's columns; the 10 folds are strictly chronological; an exact linear combination gets VIF = inf; target-aware pruning keeps Elo while classic pruning drops it; the test season is not part of development.
- `--stage test` refuses to run unless the SHA-256 fingerprint of `soccer/*.py` (except plots) and the data matches the one recorded at selection time.
- The notebook refits the saved model and reproduces the saved test metrics exactly.

## Numerical warnings
On this Mac (numpy 2.0.2 with Apple Accelerate), `divide by zero / overflow / invalid value encountered in matmul` RuntimeWarnings are emitted even for `np.random.rand(500, 8) @ np.random.rand(8)`. They are a known spurious-warning issue in that numpy/BLAS combination. Results are finite and agree with `np.einsum` and with statsmodels. The pipeline filters only that exact message.
