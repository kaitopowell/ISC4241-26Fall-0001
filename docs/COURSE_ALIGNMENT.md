# Course alignment

Checked against the user's local Lecture 5–9 Q&A and lecture notes; copyrighted course PDFs are not copied into this repository.

| Course material | Implementation and explanation |
|---|---|
| Lecture 5: preparation, missingness, encoding | Audited source fields, completion/season filters, logged warm-up exclusions, numeric pre-match inputs; no fictitious imputation |
| Lecture 6: training/validation/test, generalization, resampling | Ten expanding-window folds, same-date grouping, fixed final season, train/validation gaps, week-block bootstrap |
| Lecture 7: classification assessment | Majority and class-prior baselines, confusion, recall/precision/F1, ROC/AUC, multiclass Brier, log loss and calibration |
| Lecture 8: multicollinearity, selection, regularization | Full Pearson heatmap, exact dependence checks via VIF, fold-local pruning and L2 regularization; don't transfer OLS error assumptions blindly |
| Lecture 9: logistic regression and interpretation | Three-outcome multinomial extension, probabilities, penalized maximum likelihood, standardized class coefficients, balanced-class sensitivity |
| Team's stated comparison | Decision tree and random forest as nonlinear challengers; depth/leaf-size controls and validation rather than a presumed winner |

Only Lecture 5–9 materials were available locally. Decision tree/random forest coverage in later lectures was not independently verified; they are explicitly requested by the user/team. Elo is explained as a supplemental sports feature and tested by ablation, not presented as required course content. AIC/BIC, Wald/Score/LR tests are not forcibly added where regularization, model families or dependence make direct use misleading.
