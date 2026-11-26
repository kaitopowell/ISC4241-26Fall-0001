# Results and presentation talking points

## Why the earlier attempts looked "low" (56%, then 54%)

They predicted **three outcomes**: home win, draw or away win. On the 2024-25 season, the bookmakers' favourite was right only 54.4% of the time, so roughly 55% is the practical ceiling for that task. Draws are 25% of matches, but a draw is almost never the single most likely result. Every model, the bookmakers included, has draw recall 0.

The redo makes the main task **binary: home win vs not**. This is the Lecture 9 binary logistic regression setting. Lecture 7's sensitivity, specificity, ROC and AUC apply directly, and accuracy becomes informative.

## Numbers

**10-fold chronological CV, 2016-17 to 2023-24 (2,924 matches, 31 configurations)**

| Model | CV accuracy | Log loss | AUC |
|---|---:|---:|---:|
| Majority | 55.3% | - | 0.500 |
| Baseline logistic (5-game form) | 61.6% | 0.647 | 0.658 |
| Best decision tree (depth 2) | 65.5% | 0.630 | 0.687 |
| Best random forest (depth 4) | 66.4% | 0.614 | 0.715 |
| **Selected: compact logistic, C = 0.03** | **66.2%** | **0.609** | **0.722** |
| Bookmakers (benchmark) | 68.5% | 0.600 | 0.733 |

**Held-out 2024-25 (366 matches; evaluated once)**

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|---:|
| Majority | 58.7% | 0.00 | 1.00 | 0.500 |
| Baseline logistic | 62.6% | 0.49 | 0.72 | 0.631 |
| **Selected logistic** | **65.6%** | **0.60** | **0.70** | **0.713** |
| Random forest (not selected) | 66.9% | 0.56 | 0.74 | 0.715 |
| Bookmakers | 66.4% | 0.56 | 0.74 | 0.729 |

Paired week-block bootstrap, 95% CI:
- vs majority: +6.8 pts [+0.3, +13.5]
- vs baseline logistic: +3.0 pts [-2.1, +8.1]
- vs bookmakers: -0.8 pts [-4.3, +2.7]

So the model is significantly better than guessing, and statistically level with the betting market.

## Diagnostic → refinement story (the technical section)

| Diagnostic | Finding | Refinement |
|---|---|---|
| Correlation / VIF | `gd20 = gf20 - ga20` exactly (VIF = inf); shot share VIF 113 | VIF pruning (threshold 5), refit inside each fold |
| Binned residuals vs `elo_diff` | Upward trend: classic pruning had dropped Elo | Target-aware pruning plus a compact set that keeps Elo (final VIF: Elo 5.7, others 1.0-3.2) |
| Binned residuals vs fitted | All bins within ±2 SE | No polynomial terms needed |
| statsmodels p-values | `rest_diff` p = 0.999 | Dropped |
| Leverage / Cook's D | Max D = 0.004; the top cases are real upsets | Nothing removed |
| Residuals by season | 2020-21 over-predicted (no crowds) | `no_crowd` indicator |
| Train vs validation accuracy | Deep trees overfit | Shallow trees and large leaves for forests |

## Honest caveats to say out loud

- The forest scored 1.3 pts higher on the test season, but the model was locked on CV first. Switching now would be choosing by test score.
- The gain over the baseline logistic has a CI that includes 0. One season of 366 matches carries about ±5 pts of sampling noise.
- The bookmakers' small remaining edge comes from information we do not have: injuries, line-ups and transfers.
- Bookmaker odds are a benchmark only. Using them as inputs would mostly copy the market.

## Interpretation (odds ratio per 1 SD)

`elo_diff` 1.58 · `shots20_diff` 1.20 · `venue_ppg_diff` 1.07 · `gf5_diff` 1.02 · `ga5_diff` 0.95 · `no_crowd` 0.90 · `shots_against20_diff` 0.79
