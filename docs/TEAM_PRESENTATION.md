# Team ownership and 12-minute presentation

Assume four members; reassign names as needed. Each person creates their own slides from these verified outputs. Aim for 11 minutes; all must attend and slides must be submitted before the assigned presentation date. The supplied schedule places groups 2/4/6/8/10 on December 2, 2026; confirm your group rather than assuming a date.

| Owner | Work to understand and verify | Present | Time |
|---|---|---|---:|
| Member 1 | Source provenance, coverage, target H/D/A and genuine prediction time | Research question; source; why we excluded incomplete 2024–25 | 2:00 |
| Member 2 | Cleaning counts, five-match warm-up, form-row example, date splits, class balance | Preparation and exploration; why no random split | 2:30 |
| Kiro | Full heatmap/VIF, leakage repair, initial vs refined logistic, tuning, calibration and residual diagnostics | Main technical section: evidence → refinement → validation result | 4:00 |
| Member 4 | Independently check metrics, test comparisons, draw mistakes, uncertainty and limitations | Held-out results, conclusion, practical limits | 2:30 |

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
Member 1 checks a few fixture scores and source IDs; Member 2 hand-computes one five-match feature row; Kiro reproduces one fold and reads the VIF removals; Member 4 independently recomputes metrics from test_predictions CSV. All review reports/final_report.html and rehearse below 12 minutes. See reports/RESULTS.md for actual values after the run.
