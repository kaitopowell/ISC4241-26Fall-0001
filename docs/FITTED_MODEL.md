# The fitted model

L2-penalised logistic regression (C = 0.03), fitted on all 2016-17 to 2023-24 matches. This is the model that was locked and scored once on 2024-25.

Saved object: `artifacts/final_model.joblib` (key `selected`; the comparator models are stored under the other keys).

## Equation (raw units)

P(home win) = 1 / (1 + e^(−z)), where

```
z = −0.1988
    + 0.003002 · elo_diff
    + 0.049884 · shots20_diff
    − 0.062402 · shots_against20_diff
    + 0.080446 · venue_ppg_diff
    + 0.022756 · gf5_diff
    − 0.056715 · ga5_diff
    − 0.298443 · no_crowd
```

The model predicts "home win" when P ≥ 0.5.

## Coefficients

| Predictor | Meaning (home minus away) | Coef per 1 SD | Odds ratio per SD | Raw coef | Odds ratio per unit | SD |
|---|---|---:|---:|---:|---:|---:|
| `elo_diff` | Elo rating difference | +0.460 | 1.58 | +0.003002 | 1.003 per Elo point (1.35 per 100) | 153.1 |
| `shots20_diff` | shots per game, last 20 | +0.179 | 1.20 | +0.0499 | 1.051 per shot | 3.60 |
| `shots_against20_diff` | shots conceded per game, last 20 | −0.233 | 0.79 | −0.0624 | 0.940 per shot | 3.73 |
| `venue_ppg_diff` | home team's home PPG − away team's away PPG, last 10 | +0.066 | 1.07 | +0.0804 | 1.084 per point | 0.82 |
| `gf5_diff` | goals scored per game, last 5 | +0.022 | 1.02 | +0.0228 | 1.023 per goal | 0.98 |
| `ga5_diff` | goals conceded per game, last 5 | −0.049 | 0.95 | −0.0567 | 0.945 per goal | 0.86 |
| `no_crowd` | 1 = behind closed doors | −0.107 | 0.90 | −0.298 | 0.74 (yes vs no) | 0.36 |
| intercept | | −0.217 (standardised) | | −0.1988 (raw) | | |

The raw coefficients are the standardised ones divided by each predictor's training SD. The coefficients are shrunk by the L2 penalty, so they are smaller than an unpenalised fit would give. That is intended.

## Use it in Python

```python
import joblib
from soccer.data import load_matches
from soccer.features import build_features

model = joblib.load("artifacts/final_model.joblib")["selected"]
feats, _ = build_features(load_matches())
p_home_win = model.predict_proba(feats)[:, 1]
```

The pipeline selects its 7 columns and standardises them on its own, so pass it the full feature table.
