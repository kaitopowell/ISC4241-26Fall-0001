# Explaining the accuracy follow-up

The user asked for better than the original 56.5% accuracy. We tested more data and richer pre-match features, not a more favorable definition of success.

## What actually happened

- Training expanded from 695 rows in the original experiment to 2,985 development rows in the follow-up. Two earlier seasons initialize feature history.
- Ten chronological folds compared 42 configurations. The original recipe scored 54.28% OOF accuracy; the selected logistic model with longer form, historical shots and VIF pruning scored 55.34%.
- The selected seven inputs have final VIFs between 1.01 and 4.90. This is a real diagnostic-led refinement: the initial eleven-input model had VIFs up to 14.44.
- On 375 eligible matches in a new 2024–25 season, the selected model got 202 correct (53.87%) versus 197 (52.53%) for the refitted old recipe and 154 (41.07%) for the class-prior baseline.
- Five additional correct predictions are not strong evidence of a dependable advantage. The approximate paired week-block 95% interval for the gain is -1.33 to +4.30 percentage points.
- The model still has zero draw recall under argmax classification. More matches and richer inputs did not solve that limitation.

The old 56.5% and new 53.9% refer to different seasons. Do not advertise either their difference or a CV-to-test difference as a controlled improvement. The meaningful paired comparison is 53.9% versus 52.5% on the same new-season matches. The new season was chosen before test scores were calculated; it was not changed after seeing them.

## How to present this well

Keep the existing five-person presentation ownership. In the technical section, show the progression: small-data model → expanded history and shots → high VIF diagnosis → fold-local pruning → locked fresh-season evaluation. Conclude that the gain is modest and uncertain. Explain that a complicated forest did not beat logistic on the chosen validation criterion. Reporting a failed or modest refinement accurately is stronger methodology than selecting a model using its test score.

Use reports/expanded/figures/comparison.png, full_correlation.png, fold_diagnostics.png and fresh_test_confusion.png. Keep the original and draw-focused results labeled as separate experiments with different objectives. The expanded report gives the full table and uncertainty interval.

## What would justify another improvement cycle

New pre-match information such as player availability, lineups and independently sourced team-strength measures could add signal; it would require reliable timestamps and another evaluation set reserved before experimentation. Do not quietly use current-match shots, betting outcomes, final league tables, or remove draws to inflate accuracy. No further model tuning was done after opening the 2024–25 test.

## Verification

The extended tests check source-season completeness, future shot/score isolation, training-only imputation and VIF pruning after imputation. Existing leakage and draw-refinement tests remain. Saved probabilities and accuracy/log loss are independently verified in the report generator. Local numerical-library RuntimeWarnings persist as documented in VALIDATION.md; finite probabilities and actual numeric results are checked rather than suppressing warnings.
