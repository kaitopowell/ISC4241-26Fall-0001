## Premier League prediction: final results

Research question: can a random forest beat multinomial logistic regression on home/draw/away probabilities from pre-match team history?

The model selected before opening the test season was **logistic_refined_C0.01**. Its held-out accuracy is **56.5%**, with log loss **0.9472**. Random forest test accuracy is **56.2%**, with log loss **0.9667**. Lower log loss is better. These are observed comparisons on one archived season, not proof of universal superiority.

There is no promise of a particular grade or live forecasting accuracy. The contribution is an auditable, leakage-controlled supervised-learning experiment.

## Data and exact cleaning decisions

The source is TheSportsDB data already saved in the team's original Premier League notebook. No API key or database is required. The saved retrieval timestamp is unknown. We extracted 1,520 scheduled events, excluded 161 unfinished events and 219 completed events from the incomplete 2024–25 season. The three complete seasons contain 1,140 matches.

Five prior recorded matches are required for both teams. This excludes 75 warm-up rows from modeling while retaining them in team histories: 695 development rows (2021–22/2022–23) and 370 held-out rows (2023–24). This means the results do not cover every promoted-team fixture. No mean imputation or outlier deletion was needed for these selected inputs. Optional API statistics were not used. Valid extreme scorelines remain; every exclusion and flagged scoreline is recorded in CSV.

The original notebook added the current result before calculating a last-five feature. Our replacement constructs every match's inputs before adding that day's results. A test changes all current/future scores and confirms earlier/current predictors stay unchanged.

![01_class_counts](figures/01_class_counts.png)

![02_distributions](figures/02_distributions.png)

## Features and collinearity

We begin with home/away five-game points, goals for, goals against, goal difference, win fraction, capped rest days and prior Elo difference. Elo is a supplemental team-strength feature, not claimed as a taught model. Its constants were fixed and a no-Elo comparison is included.

The full heatmap shows points and wins are strongly correlated. More importantly, goal difference equals goals for minus goals against: full-model VIFs are infinite. Pairwise correlation alone would miss this exact multivariable dependency.

Refinement removes goal difference, uses home-minus-away differences, and iteratively removes predictors with VIF above 5 using only each training fold. The final logistic features are goals-for difference, goals-against difference, win-fraction difference, rest difference and Elo difference. Final VIFs range from about 1.00 to 3.51. This improves interpretability without claiming independent or causal predictors.

![03_full_correlation](figures/03_full_correlation.png)

![09_final_correlation](figures/09_final_correlation.png)

## How validation stays honest

Ten expanding chronological folds use strictly earlier dates for training and later dates for validation. Same-day fixtures stay together. The first 266 rows initialize training; 429 distinct later development fixtures receive out-of-fold predictions. Scaling and VIF screening are refitted inside each fold. Model parameters stay fixed within a validation block, while newly observed results may update later pre-match features, exactly as sequential forecasts would.

Nineteen configurations compare majority/prior baselines, initial and refined logistic regression, a class-weighted logistic model, a no-Elo logistic model, decision trees and random forests. The primary selection criterion is pooled out-of-fold multiclass log loss. C controls inverse L2 regularization; tree depth and leaf size control complexity. CV is also used for tuning, so it is not an unbiased nested-CV estimate. The untouched newest complete season supplies the final assessment.

![06_cv_timeline](figures/06_cv_timeline.png)

![07_model_comparison](figures/07_model_comparison.png)

![08_overfitting](figures/08_overfitting.png)

## Diagnostic evidence and refinement decisions

Initial logistic OOF loss was 1.0023; refined logistic loss is 0.9838. This combined change includes difference encoding, VIF screening and stronger regularization; do not attribute all improvement to VIF alone. Refined logistic and forest validation losses differ by only 0.0014, so their ranking is close.

The confusion matrix reveals zero draw recall for the selected OOF model. Its draw probabilities can still be useful even when draw never has the largest probability. Class weighting increases OOF draw recall to 15.3%, but worsens log loss to 1.0219. We keep the probability-focused winner and report the limitation. A different decision objective could justify a different choice, but must be specified before test evaluation.

Calibration and binned probability residuals reveal underprediction of some home-win probabilities and overprediction of some away-win probabilities. The extreme home-win bin has only two OOF examples: do not treat its shape as reliable. We do not fit an extra calibrator to these same OOF results and claim independent improvement. Monthly loss varies, showing that performance is not constant through time. No OLS normal-residual assumption is imposed on a categorical response.

At matched C=0.1, logistic OOF loss is 0.9841 with Elo versus 1.0257 without it. This is evidence for added predictive information in this experiment, not a causal effect.

![11_refined_oof_confusion](figures/11_refined_oof_confusion.png)

![11_refined_oof_calibration](figures/11_refined_oof_calibration.png)

![11_refined_oof_residuals](figures/11_refined_oof_residuals.png)

![11_refined_oof_roc_pr](figures/11_refined_oof_roc_pr.png)

## Held-out season: no model changes afterward

| Model | CV log loss | Test log loss | Test accuracy | Test macro F1 | Test draw recall |
| --- | --- | --- | --- | --- | --- |
| logistic_refined_C0.01 | 0.9838 | 0.9472 | 0.5649 | 0.4131 | 0.0000 |
| forest_d8_leaf25 | 0.9853 | 0.9667 | 0.5622 | 0.4132 | 0.0000 |
| logistic_initial | 1.0023 | 0.9311 | 0.5676 | 0.4243 | 0.0125 |
| tree_d2_leaf40 | 1.0166 | 0.9961 | 0.5514 | 0.4083 | 0.0000 |
| logistic_balanced | 1.0219 | 0.9798 | 0.5243 | 0.4631 | 0.1625 |
| logistic_no_elo | 1.0257 | 0.9884 | 0.5676 | 0.4113 | 0.0000 |
| prior | 1.0567 | 1.0542 | 0.4595 | 0.2099 | 0.0000 |
| majority | 18.9040 | 19.4831 | 0.4595 | 0.2099 | 0.0000 |

The selected model's macro one-vs-rest AUC is 0.6765, multiclass Brier score is 0.5592, and draw recall is 0.0%. Accuracy should always be shown alongside these class-sensitive and probability metrics. The class-prior baseline accuracy is 45.9%.

Selected-minus-prior mean test log loss is -0.1070. The approximate paired week-block bootstrap 95% interval is [-0.1453, -0.0688]; negative values favor the selected model. Two thousand resamples preserve within-week blocks, but cross-week/team dependence may remain. This is conditional on the fitted models and does not establish a significant logistic-versus-forest difference.

The initial logistic model actually has lower test log loss (0.9311) than the selected refined model (0.9472). It also has slightly higher test accuracy. This is a useful example of validation rankings changing on new data; refinement did not improve every held-out metric. We retain the CV-selected model rather than choosing retrospectively.

The same preselected family representatives are reported for transparency. The selected model is not changed after inspecting this table. All results concern sequential forecasts within 2023–24 using a model fitted through 2022–23, not current 2026 fixtures.

![20_test_confusion](figures/20_test_confusion.png)

![20_test_calibration](figures/20_test_calibration.png)

![20_test_roc_pr](figures/20_test_roc_pr.png)

![20_test_residuals](figures/20_test_residuals.png)

## Interpretation and limits

The coefficient heatmap describes standardized inputs in multinomial logistic regression. To compare home versus away log odds, subtract their coefficients; an individual coefficient is not a percentage-point probability effect. The shallow decision tree illustrates nonlinear rules; its thresholds are on standardized values.

Repeated teams, one test season, stale historical data, promoted-team warm-up exclusions, narrow draw probabilities and absent lineup/injury/transfer information limit generalization. Valid surprises are not deleted. Further improvements require fresh independent seasons, not repeated tuning on this test season.

The original Flask pages and saved pickles are retained as legacy work, not the evaluated model. Use the new historical prediction command for the demonstration. It rejects unsupported future dates. See docs/METHODOLOGY.md, docs/COURSE_ALIGNMENT.md and docs/TEAM_PRESENTATION.md for details and rehearsal questions.

![15_logistic_coefficients](figures/15_logistic_coefficients.png)

![16_tree](figures/16_tree.png)
