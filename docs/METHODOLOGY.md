# Methodology and decisions to defend

## Research question
Can a random forest improve Premier League home/draw/away probability predictions over multinomial logistic regression, using the same pre-match information? Three outcomes require **multinomial**, not ordinary binary logistic regression. Retain draws; do not silently convert them to losses.

## Scope and provenance
Use the original team's cached TheSportsDB league 4328 event lists in the Premier League notebook. Parse saved stream outputs with `ast.literal_eval`; never execute its API/database cells or deserialize its old models. Record source SHA-256 and extraction cells. The snapshot retrieval timestamp was not recorded by the original authors, so we do not invent one. The snapshots include 380 completed fixtures each in 2021–22, 2022–23 and 2023–24; 2024–25 is incomplete and excluded. This is an archived experiment, not a current 2026 forecasting service. TheSportsDB is the original source; availability in the repository does not establish an unrestricted redistribution license.

## Cleaning
Filter to league 4328, reject conflicting match IDs, deduplicate identical records, retain completed matches with valid dates/teams/nonnegative integer scores. Require 20 clubs, 380 unique home-away pairings and 19 home/19 away games per club for each included season. Preserve valid unusual results. `reports/cleaning_audit.csv` counts each sequential exclusion; `season_coverage.csv` explains incomplete-season exclusion. `unusual_scorelines_development.csv` flags 7+ total goals or 5+ goal margins for inspection, not automatic removal. These are plausible archived scorelines, not independently externally verified results.

Avoid the legacy notebook's complete-case deletion across many optional match statistics: it removed many fixtures, corrupting the meaning of 'last five'. Here all completed fixtures update team history, including warm-up fixtures that are not eligible for supervised modeling. A modeled row requires five previous matches for **both** teams; the exclusions are saved individually.

## Feature availability
For each match date, build all rows BEFORE updating any team history for that date. This is conservative because exact completion times are not reliably established. Features combine home and away performances in each team's previous five recorded league fixtures, carrying history across seasons. Newly promoted clubs have no lower-league history here and need a five-match warm-up. We do not impute their outcomes. Earlier results in a validation/test block update later pre-match features; model parameters stay fixed for that block. This is a sequence of pre-match forecasts, not a whole-season forecast made at its start.

Initial inputs: each team's mean points, goals for, goals against, goal difference, win fraction and capped days since last game; plus Elo rating difference. Rest is capped at 14 days by a predeclared rule to limit off-season discontinuities. Elo is an **optional domain feature, not claimed as a course-taught learning method**: initial 1500, K=20, home advantage 60, logistic expected-score scale 400. All fixed, never optimized on the test set. The no-Elo logistic ablation measures its contribution. Rating does not encode injuries, transfers, lineups or team finances.

Refinement: use home-minus-away differences (points, goals for, goals against, win fraction, rest) plus Elo difference; drop goal difference because it equals goals for minus goals against. Home venue remains implicit in target orientation and logistic intercept. A fold-local VIF transformer removes the largest VIF until all remaining VIFs are <=5 (or one feature remains). This is an explicit heuristic, not proof of optimal features. Its removal log is saved for every model/fold. The same reduced information and transform are offered to tree and forest so the comparison is controlled. Trees do not require low VIF for valid prediction, but correlated variables can obscure their importance.

## What was wrong in the legacy analysis?
1. Cell 14 adds the current winner to its history **before** computing last-five wins: direct target leakage. We do not reuse those columns or accuracy claims.
2. The draw string is compared against `None` in streak logic. Our new points update explicitly assigns one point per team for a draw.
3. Optional-statistic complete-case filtering discarded matches before forming history.
4. No reproducible comprehensive model comparison or locked test workflow; notebook results and serialized web models are not interchangeable with this experiment.
5. The existing Flask demo needs a database and stale external assets. It remains a clearly labeled legacy interface; use the new offline prediction command and notebook for assessed work.

## Validation and selection
Development: eligible 2021–22 and 2022–23 fixtures. Test: eligible 2023–24 fixtures, selected because it is the newest **complete season in the snapshot**. This is not the newest season in today's world. Within development, start with roughly 40% of match dates and evaluate on 10 successive nonoverlapping date blocks using expanding training sets. Same-day matches never straddle a boundary. Every train and validation fold must contain all three outcomes; the runner fails rather than silently evaluating invalid folds.

Select by pooled out-of-fold multiclass log loss, breaking an exact tie by macro F1. Log loss rewards honest probabilities and penalizes confidently wrong predictions; macro F1 and draw recall expose majority-class dominance. The primary metric and candidate grid are fixed before test evaluation. Ten folds are a compromise for this data volume, not an automatic guarantee of reliability. CV is used for tuning AND comparison, so its winning score is selection-optimistic; it is **not nested CV**. Final test is the generalization estimate. Comparing several prespecified families on test is descriptive; do not switch the chosen model after seeing the table.

All scaling and VIF selection are fitted only on each fold's training portion. No SMOTE: synthetic fixture records are hard to justify; class-weighted logistic regression tests the recall/probability trade-off instead. Hyperparameter search includes logistic C, tree depth/minimum leaf size and forest depth/minimum leaf size. C is inverse L2 regularization, not the coefficient itself. Stochastic estimators use seed 42 and forests 150 trees.

## Metrics
- Accuracy: fraction of predicted labels correct (argmax probability).
- Macro F1: average F1 with equal weight for away/draw/home; zero division set to zero.
- Balanced accuracy: mean recall across these three classes.
- Multiclass log loss: mean negative log probability of the observed class.
- Multiclass Brier: mean **sum across classes** of squared probability errors, range 0–2. Not the binary 0–1 convention.
- Macro one-vs-rest AUC: average class-vs-rest ranking AUC. Plot three ROC and PR curves, not a single binary curve mislabeled as multiclass.
- Draw recall: draws correctly identified divided by actual draws.
- Probability baseline: training class frequencies. Majority baseline has hard one-class probabilities and consequently very poor log loss; it remains a useful accuracy benchmark.

A paired week-block bootstrap summarizes selected-minus-prior test log loss (2,000 resamples, seed 42). It is an approximate conditional uncertainty analysis, not an independent-sample guarantee or causal result; teams and neighboring weeks can remain dependent.

## Diagnostics and actual refinement
Initial-window full-feature VIF detects exact dependencies before tuning: goal difference = goals for - goals against. Difference encoding and fold-local VIF screening address redundancy, then regularization is tuned. Compare initial and refined OOF log loss, calibration, draw recall, confusion and training/validation gap; a cleaner model is not automatically more accurate. The retained-feature correlation heatmap and VIF table verify what changed.

Use classification-appropriate diagnostics: class-specific reliability curves with bin support, binned indicator-minus-probability residuals, class confusion, ROC/PR, loss through time, and train-versus-validation loss. Sparse calibration bins are uncertain. Inspect high-loss matches without deleting legitimate surprises. Extreme scorelines are predictor-history inputs for future matches, not regression residuals to trim.

Do **not** impose OLS normal residuals, constant variance, ordinary Q–Q diagnostics, or Cook's distance on a three-class outcome and call that logistic model validation. Do not use ordinary OLS AIC/BIC to rank random forests. We omit formal coefficient p-values because regularization, selection and repeated-team dependence undermine naive inference. Multinomial coefficients are per standardized input and compare class log odds via coefficient differences, not a direct probability change. Correlation and fitted effects are not causation.

## Source links
- https://www.thesportsdb.com/documentation
- https://scikit-learn.org/1.6/modules/cross_validation.html
- https://scikit-learn.org/1.6/modules/calibration.html
- https://scikit-learn.org/1.6/modules/linear_model.html#logistic-regression
