# Team ownership and 12-minute presentation

Five members, confirmed by Kiro. Each person creates their own slides from the verified outputs. Aim for 11 minutes, leaving one minute of buffer; all must attend and slides must be submitted before the assigned presentation date. The supplied schedule places groups 2/4/6/8/10 on December 2, 2026; confirm your group.

| Owner | Work to understand and verify | Present | Time |
|---|---|---|---:|
| Member 1 | Source provenance, research question and target H/D/A | Problem, motivation, source and archived-data limits | 1:30 |
| Member 2 | Cleaning counts, warm-up exclusions, pre-match feature example and class balance | Data preparation and exploration | 2:00 |
| Member 3 | Logistic probabilities, tree/forest reasoning and regularization | Baselines, candidate models and why each was tested | 2:00 |
| Kiro | Leakage repair, full heatmap/VIF, nested validation, draw rule and failed accuracy constraint | Main technical section: evidence → refinement → honest evaluation | 3:30 |
| Member 5 | Independently check final metrics, draw errors, uncertainty and limitations | Later holdout results and measured conclusion | 2:00 |

## Slide sequence
1. Can a forest beat logistic regression before kickoff? One fixture, three probabilities.
2. Three complete cached seasons, original source, train/test timeline.
3. Cleaning flow: numbers directly from cleaning_audit.csv and insufficient_history.csv. Valid unusual scores retained.
4. Five prior matches, home-minus-away differences, one hand-worked example. Current result excluded.
5. Class balance and baselines. Why accuracy alone misses draws.
6. Kiro: full correlation matrix in appendix; main slide highlights exact goal-difference redundancy. Show before/after VIF and explain fold-local fit.
7. Kiro: 10 expanding folds and candidate models; primary metric log loss; training/validation gap.
8. Kiro: initial vs refined OOF reliability/residual diagnostic. State a measured improvement OR lack of improvement honestly. Explain what calibration does not establish.
9. Test-season metrics and confusion matrix; identify the model locked by CV, even if another looks better on this test.
10. One high-loss match, week-block interval, limitations, answer research question without overclaiming.

## Kiro's technical talking points
- Old feature computed the win count after adding the current outcome: leakage. Our unit test changes current/future scores and proves earlier features stay unchanged.
- Mean goal difference is exactly goals for minus goals against, so using all three creates a redundant design. Correlation alone can miss multivariable dependencies; VIF regresses one input on all others.
- We use multinomial logistic regression: softmax gives three probabilities summing to one. The intercept represents home/draw/away tendencies even with all form differences zero.
- L2 regularization limits coefficient size; smaller C means stronger shrinkage. We tune C on earlier data only.
- Forests model nonlinear interactions but can overfit small samples. Depth and leaf-size controls matter more than simply adding trees.
- A reliability point compares average predicted probability with observed class fraction in a bin. Bin counts show uncertainty. Brier measures probability accuracy, not calibration alone.
- A binned residual is observed class indicator minus predicted probability. A systematic offset suggests probability bias; normal residuals are not required.
- Draws can have reasonable probabilities without ever being the largest probability. This can yield low draw recall with good log loss. Balanced weights trade off objectives; report both.
- Ten folds are successive future blocks. Fold results are dependent because training windows overlap; SD bars are not confidence intervals.
- Earlier test matches can update later form because their results are available by then. No test result updates coefficients or model selection.

## Questions to rehearse
Why exclude an incomplete season? Why five prior matches? Why not random stratified folds? Why do Elo and recent form differ? What is cold start? What does VIF=5 mean? Why no SMOTE? Why no ordinary regression Q–Q test? How do multiclass AUC/Brier differ from binary versions? Why not choose the highest test accuracy after comparison? Does a high score imply profitability? (No: no odds/costs model.) Can it forecast current 2026 fixtures? (No: this is an archived snapshot.)

## Individual verification
Member 1 checks a few fixture scores and source IDs; Member 2 hand-computes one five-match feature row; Kiro reproduces one fold and reads the VIF removals; Member 3 verifies model settings; Member 5 independently recomputes metrics from the original and follow-up prediction CSV files. All review reports/final_report.html and rehearse below 12 minutes. See reports/RESULTS.md for actual values after the run.

## Updated 10-slide sequence for the draw follow-up

1. Member 1: research question and three possible outcomes (0:45).
2. Member 1: data source and archived snapshot; explain both experiments (0:45).
3. Member 2: cleaning flow and warm-up exclusions (1:00).
4. Member 2: hand-worked pre-match features and class balance (1:00).
5. Member 3: baseline multinomial logistic, tree and forest; why each (1:00).
6. Member 3: initial results and why argmax can ignore draws (1:00).
7. Kiro: leakage repair plus full correlation/VIF evidence (1:00).
8. Kiro: 10 outer × 3 inner chronological folds, model AND decision-rule tuning; explain the accuracy/draw-recall constraint (1:30).
9. Kiro: nested results and why the failed constraint prevents claiming a clear upgrade (1:00).
10. Member 5: later holdout confusion matrix, 51.4% vs 50.0% same-row accuracy, 10/57 draws caught, uncertainty and conclusion (2:00).

Use the full heatmap and model-grid tables in an appendix so the main presentation stays readable. The old 56.5% belongs to a different season; do not compare it directly with the new 51.4%. The selected forest's final draw multiplier is 1.0; do not claim threshold boosting caused its final draw predictions. Nested validation evaluated the whole search procedure, including boosted rules where selected. All outer folds and final development search missed the declared trade-off constraints, so the forest is an alternative rather than a proven accuracy upgrade.

Actual follow-up results: reports/draw_refinement/RESULTS.md. Each member should be able to explain the shared conclusion, even if another member presents the technical slide.
