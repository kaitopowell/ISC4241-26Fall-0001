# Frozen follow-up protocol

Purpose: investigate accuracy and draw recognition without silently retuning the already-opened 2023–24 test. The v1 experiment remains unchanged.

Development: all eligible complete 2021–22 through 2023–24 matches. The previously inspected 2023–24 season is now explicitly development data. Final check: eligible completed 2024–25 matches from the existing partial snapshot, never used in v1 tuning/evaluation. This is a partial-season holdout, not a full-season estimate, with snapshot/fixture-completion selection limitations.

Use 10 outer expanding date folds. In each outer training set use 3 inner expanding date folds for model AND decision-rule selection. After nested evaluation, run the identical inner selection procedure on all development data, freeze it, and evaluate the later partial season once. No candidate/grid changes based on that outcome.

Models: multinomial L2 logistic regression (C=0.1 or 1; original signed inputs or an augmented set; draw training weight 1 or 1.5), two shallow decision trees and two random forests (depth/leaf controls, balanced weights). Fixed v1 logistic C=.01 is the reference, refitted on the same training data. No new advanced learning algorithms or fabricated synthetic matches.

Augmentation: absolute prior Elo gap, total recent scoring and total recent conceding, from existing pre-match features. The signed difference alone cannot easily express 'closely matched teams draw more often.' This is a hypothesis, not an established effect. VIF selection and scaling remain inside training folds.

Decision rule: multiply the draw probability by one of [1,1.25,1.5,1.75] ONLY when choosing the label. Keep original probabilities unchanged for log loss, Brier, ROC and calibration; these decision scores are not probabilities. This is a multiclass extension of lecture threshold selection.

Selection objective: among candidates whose inner OOF accuracy is no more than 2 percentage points below the fixed reference and whose draw recall is at least 10%, maximize macro F1 (ties: accuracy, then lower log loss). If none meet both constraints, choose highest macro F1, explicitly flag the unmet constraints. The tolerances represent a declared practical trade-off, not guaranteed future performance or statistical significance.

Report nested outer predictions, draw precision/recall/F1, accuracy, macro F1, log loss, changes against a same-row reference, paired week-block intervals and calibration. Display argmax and draw-aware decisions separately. Preserve the first final result via a manifest; no retuning after final inspection. Final held-out comparisons are descriptive and conditional on this experiment, not proof of universal superiority.
