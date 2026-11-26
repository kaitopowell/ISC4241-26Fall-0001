# Feature and model guide

For a match, H means home win, D draw and A away win. The final scores create this target **after** kickoff; they are never predictor inputs for the same match.

| Input | Plain meaning | Calculation before the match |
|---|---|---|
| ppg5 | Recent points per game | Mean of last five points: win=3, draw=1, loss=0 |
| gf5 | Recent scoring | Mean goals scored in last five league games |
| ga5 | Recent conceding | Mean goals conceded in last five league games |
| gd5 | Recent goal difference | gf5 minus ga5; redundant if both already included |
| win5 | Recent win fraction | Wins in last five divided by five |
| rest | Days since last recorded league game | Capped at 14 days, not all competitions |
| *_diff | Relative recent performance | Home value minus away value |
| elo_diff | Relative accumulated strength | Home prior Elo minus away prior Elo |

Example: a home team's last five results yield points [3,1,0,3,1], goals for [2,1,0,3,2], goals against [0,1,1,1,2]. Its ppg5=1.6, gf5=1.6, ga5=1.0, gd5=0.6 and win5=0.4. If the opponent's gf5 is 1.2, gf_diff=0.4. Never include the result you are predicting among those five games. The window is the team's five games at either venue, not its five home-only games.

## Equations to explain

**Multinomial logistic regression:** class k gets score z_k = intercept_k + sum_j coefficient_kj × standardized_feature_j. Then p_k = exp(z_k) / sum_c exp(z_c). The three probabilities sum to one. L2 regularization shrinks coefficients; smaller C means stronger shrinkage. The most probable class is the default predicted label.

**Log loss:** average -log(probability assigned to the actual outcome). If the actual result is draw and the model assigned it probability 0.20, that match's loss is -log(0.20)=1.609. A confident mistake gets a large penalty.

**VIF_j = 1 / (1 - R²_j)**, where R²_j comes from regressing predictor j on the other predictors with an intercept. R²_j=0.8 implies VIF=5. An exact linear dependency gives infinite VIF. This is about redundancy in inputs, not the classification model's predictive R².

**Elo:** E_home = 1 / (1 + 10^((rating_away - rating_home - 60)/400)); update = 20 × (S_home - E_home), where S_home is 1, 0.5 or 0. Home rating increases by update and away decreases by update. Start at 1500; construct features before applying the current date's updates. Fixed constants are domain assumptions, not estimated causal quantities.

**ROC:** one class versus the other two, plot true-positive rate against false-positive rate across thresholds. Its diagonal is chance ranking; AUC summarizes ranking, not calibration. AUC does not show whether probabilities such as 0.7 occur about 70% of the time.

**Calibration:** group similar predictions, compare their mean probability to observed frequency. Here the ideal diagonal means calibrated probabilities, a different interpretation from the ROC diagonal. Sparse bins need caution.

**Bootstrap:** resample whole test-week blocks with replacement 2,000 times and recalculate the paired mean loss difference. Both models are evaluated on the same resampled fixtures. This approximates uncertainty conditional on the trained models; it does not train 2,000 independent models or remove all team dependence.
