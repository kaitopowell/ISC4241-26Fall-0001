# Validation record

Environment: macOS, Python 3.9.6, package versions in requirements-analysis.txt.

- Nine pytest tests passed: complete source seasons, conflicting duplicates, current/future result independence, same-day ordering, chronological folds, fold-local VIF, class ordering/metrics, hand-computed draw history and predictor allowlisting across logistic/tree/forest.
- All nine code cells of Final_Project.ipynb executed successfully with the local Jupyter kernel; no error outputs.
- scripts/verify_results.py independently recalculates all saved test accuracies/log losses, verifies finite probabilities summing to one, reconstructs selected-model softmax without BLAS matrix multiplication, and cross-checks VIF residual arithmetic using elementwise sums. It also exercises supported and rejected historical-demo requests.
- Development and test figures were inspected for readable labels, meaningful axes and correct class ordering.
- The selected configuration and source/code fingerprint are preserved in selection_lock.json and test_manifest.json. Exact reproduction is allowed; changed experiments cannot overwrite the held-out results through the runner.

## Numerical warning observed

This local NumPy/BLAS execution emitted divide-by-zero/overflow/invalid RuntimeWarnings from some matrix multiplications, even though the returned arrays were finite. The independent elementwise VIF and softmax calculations match to tolerances of 1e-12 (softmax absolute tolerance 1e-14), and saved metrics were independently verified. The cause of the warning was not conclusively diagnosed; it is recorded rather than silently suppressed. Re-run the numerical verifier if changing platform or dependencies. No model choice or result was changed to address the warning.

## Scope

The experiment is complete; final slides still belong to the team members, as requested in the supplied project notes. The presentation outline and images support those slides. The legacy Flask/database application has not been converted into a live forecasting service. There is no independent verification of every historical score against an external provider, nor a new current-season data feed.

## Draw-recognition follow-up

The additional four tests cover unchanged probability values under the draw decision rule, nested chronological boundaries, a separate later holdout, enforced selection constraints/fallback reporting and predictor allowlisting. Thirteen tests pass in total. The small synthetic decision-rule test triggers sklearn's warning that a predicted class is absent from that toy example's true labels; this is intentional and does not occur in the three-class evaluation cohorts.

The follow-up notebook was executed successfully. A disposable-directory reproduction reran the entire nested search and later evaluation; all six audited tables (nested summary/predictions, outer splits, final metrics/predictions and VIF) matched within numerical tolerance. Original v1 results and source files remained unchanged. The follow-up trade-off and confusion figures were visually inspected.
