"""Generate and execute Home_Win_Prediction.ipynb (the course submission notebook)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Predicting Premier League home wins
**ISC 4241 final project.** Can a model built only from pre-match information predict whether the home team wins?

Plan, following the course:
1. Data preparation and leakage control (Lecture 5)
2. Training / validation / test design with **10-fold chronological cross-validation** (Lecture 6)
3. **Baseline logistic regression** vs refined logistic, decision trees and random forests (Lectures 7 and 9, plus trees)
4. **Collinearity**: correlation heatmap and VIF (Lecture 8)
5. **Diagnostic plots**: binned residuals, leverage and Cook's distance, residuals by season, train vs validation gap. These drive the refinements.
6. One evaluation of the locked model on the **untouched 2024-25 season**, scored with accuracy, sensitivity, specificity, ROC/AUC and log loss (Lecture 7)

The heavy lifting is in the `soccer/` package; this notebook runs and explains it."""),
code("""import json, warnings
warnings.filterwarnings('ignore', message='.*encountered in matmul')   # spurious macOS numpy warning, see docs
import numpy as np, pandas as pd, joblib
from IPython.display import Image, display
from soccer.data import load_matches
from soccer.features import build_features, ALL_FEATURES
from soccer.models import vif_table, binary_metrics, bookmaker_prob
pd.set_option('display.width', 160); pd.set_option('display.precision', 4)
R, F = 'reports/', 'reports/figures/'"""),
md("## 1. Data\nEleven complete seasons from Football-Data.co.uk. Every season is checked for 380 matches, 20 teams, no duplicate fixtures, and results that match the scores."),
code("""matches = load_matches()
print(matches.groupby('season').size().to_string())
matches[['date','home','away','hg','ag','result','hs','as_','hst','ast']].head()"""),
md("""## 2. Pre-match features (no leakage)
For each match we use only **earlier dates**. All of a day's matches are predicted first, then that day's results enter the history.
Features are mostly *home minus away* differences:

| Feature | Meaning |
|---|---|
| `elo_diff` | Elo rating difference (updated after every match, goal-margin weighted, regressed between seasons) |
| `ppg5/gf5/ga5_diff` | last-5-match points, goals for and against (the **original project's** features = baseline) |
| `*20_diff` | last-20-match averages: points, goals, goal difference, shots, shots on target, shot-on-target share |
| `venue_ppg/gd_diff` | home team's last 10 *home* games vs away team's last 10 *away* games |
| `rest_diff` | days since the last league match (capped at 14) |
| `no_crowd` | 1 for matches played behind closed doors (June 2020 - May 2021) |

Bookmaker odds are kept **only as a benchmark** and never enter a model."""),
code("""feats, excluded = build_features(matches)
dev  = feats[(feats.season >= '2016-2017') & (feats.season <= '2023-2024')].reset_index(drop=True)
test = feats[feats.season == '2024-2025'].reset_index(drop=True)
print(f'development: {len(dev)} matches, test: {len(test)} matches, warm-up exclusions: {len(excluded)}')
print('home-win rate  dev %.3f  test %.3f' % (dev.home_win.mean(), test.home_win.mean()))
dev[['match_id'] + ALL_FEATURES + ['home_win']].head()"""),
md("**Leakage check.** Change the current match's score and shots to absurd values. Its own features must not move. (The same check runs in `tests/test_pipeline.py`, along with future-match and same-day checks.)"),
code("""target = feats.match_id.iloc[1500]
changed = matches.copy(); i = changed.index[changed.match_id == target][0]
changed.loc[i, ['hg','ag','hs','as_','hst','ast','result']] = [9, 0, 40, 0, 30, 0, 'H']
a = feats.set_index('match_id').loc[target, ALL_FEATURES]
b = build_features(changed)[0].set_index('match_id').loc[target, ALL_FEATURES]
print(target, '-> features unchanged:', np.allclose(a.astype(float), b.astype(float)))"""),
display_fig := code("display(Image(F + '01_home_win_rate_by_season.png'))"),
md("""## 3. Why a binary target
Draws are about 25% of matches, but a draw is almost never the single most likely outcome. That caps three-way accuracy near 55%:
the bookmakers, who have far more information, picked the right outcome only ~54% of the time in 2024-25.
"Home win vs not" is the Lecture 9 binary logistic setting. Sensitivity, specificity and ROC apply directly, and accuracy is more informative.
The three-way result is still reported in section 8."""),
md("## 4. Collinearity: correlation and VIF\n$VIF_j = 1/(1-R_j^2)$, where $R_j^2$ comes from regressing predictor $j$ on all the others."),
code("""display(Image(F + '02_correlation_heatmap.png'))
v = vif_table(dev[ALL_FEATURES]); v.round(2).to_frame('VIF')"""),
md("""`gd20 = gf20 - ga20` **exactly**, so its VIF is infinite (perfect collinearity: $X^TX$ is singular). The shot-share variables have VIF > 100.
The classic fix (drop the highest VIF until all are at most 5) removes **`elo_diff`**, our strongest predictor. The diagnostics below show why that is a problem."""),
code("display(Image(F + '03_vif_before_after.png'))"),
md("## 5. Diagnostic plots (initial pruned logistic model, fitted with statsmodels)"),
code("""display(Image(F + '04_logistic_diagnostics.png'))
print(open('reports/logit_statsmodels_summary.txt').read()[-1700:])"""),
md("""**Reading the diagnostics, and what we changed:**
- *Binned residuals vs `elo_diff`* slope upward: negative for weak home teams, positive for strong ones. The model is missing team-strength signal, because pruning threw away Elo.
  → **Refinement 1:** *target-aware* VIF pruning (among VIF > 5 predictors, drop the one least correlated with the outcome), plus a **compact** set that keeps Elo.
- *Binned residuals vs fitted probability* all fall within ±2 SE: no sign of a wrong link or curvature, so no polynomial terms.
- *Leverage vs residual*: the maximum Cook's D is 0.004. Some matches pass the strict 4/n ≈ 0.0014 screen, but none comes near the 0.5-1 level of real influence. The top ones are genuine upsets (e.g. Leicester 1-0 Liverpool, Dec 2021), not data errors, so nothing was removed.
- `rest_diff` has p = 0.999. → **Refinement 2:** dropped from the compact model."""),
code("""display(Image(F + '05_oof_residual_by_season.png'))
display(Image(F + '06_tree_forest_train_vs_valid.png'))"""),
md("""- *Residuals by season*: 2020-21 (no crowds) is over-predicted. Home advantage almost vanished without fans, hence `no_crowd`.
- *Training vs validation*: deeper trees and forests gain training accuracy without gaining validation accuracy, which is overfitting (bias-variance).
  → **Refinement 3:** forests limited to shallow trees with large leaves on the compact set."""),
md("""## 6. 10-fold chronological cross-validation
Shuffled k-fold would let the model train on 2023 matches to predict 2018 matches. Instead the development period is cut into date-ordered blocks:
fold *k* trains on everything before block *k* and validates on block *k*. Every model, and every VIF pruning step, is refit inside each fold.

**Selection rule (declared before refining):** highest pooled out-of-fold accuracy, with models within 0.25 pts treated as tied and the lower log loss winning."""),
code("""cv = pd.read_csv(R + 'cv_all.csv')
book = json.load(open(R + 'bookmaker_cv.json'))
print(f"bookmaker benchmark (CV period): accuracy {book['accuracy']:.4f}, AUC {book['auc']:.4f}")
cv[['model','family','stage','accuracy','log_loss','auc','sensitivity','specificity','fold_acc_sd','train_accuracy']]"""),
code("""display(Image(F + '07_cv_model_comparison.png'))
display(Image(F + '08_cv_accuracy_by_fold.png'))
display(Image(F + '09_oof_calibration_roc.png'))"""),
code("""lock = json.load(open(R + 'selection_lock.json')); print(json.dumps(lock, indent=2))
print(vif_table(dev[lock['features']]).round(2).to_string())
display(Image(F + '10_selected_coefficients.png'))
pd.read_csv(R + 'selected_coefficients.csv', index_col=0)"""),
md("""**Selected model:** L2-penalised logistic regression (C = 0.03) on 7 predictors. It ties the best random forest on accuracy and has the best log loss and AUC of all 31 configurations.
`elo_diff` has VIF 5.7, slightly above 5 and below the usual cutoff of 10. We kept it because the residual plots showed that dropping it hurt, and the L2 penalty stabilises its coefficient.

*Interpretation:* one standard deviation of Elo advantage multiplies the odds of a home win by about 1.58. Conceding more shots lowers them (odds ratio 0.79 per SD)."""),
md("## 7. The held-out test: 2024-25\nThe locked models were fitted on all 2016-24 data and scored **once** on 2024-25. Below we refit the saved selected model and confirm it reproduces the saved numbers."),
code("""bundle = joblib.load('artifacts/final_model.joblib')
p = bundle['selected'].predict_proba(test)[:, 1]
print('re-computed:', {k: round(v, 4) for k, v in binary_metrics(test.home_win, p).items()})
res = pd.read_csv(R + 'test_results.csv'); res"""),
code("""display(Image(F + '11_test_confusion_roc.png'))
boot = json.load(open(R + 'test_bootstrap.json'))
for k, (pt, lo, hi) in boot.items():
    print(f'{k:36s} {pt:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]')"""),
md("""**Results on 2024-25 (366 matches):**
- Selected logistic: **65.6%** accuracy, AUC 0.713, sensitivity 0.60, specificity 0.70.
- That is +3.0 pts over the baseline logistic (62.6%) and +6.8 pts over always guessing "not a home win" (58.7%). The interval for the gain over majority excludes zero; the interval for the gain over the baseline logistic includes it.
- Bookmakers: 66.4%. The difference is -0.8 pts (CI -4.3 to +2.7). **Using only public match statistics, our model is statistically level with the betting market.**
- The best random forest scored 66.9% on this season. It was not selected, because the choice was locked on cross-validation. Switching after seeing the test score would be selecting on the test set."""),
md("## 8. Secondary: three-way home/draw/away\nSame locked predictors, multinomial logistic, no further tuning."),
code("display(pd.read_csv(R + 'three_way_test.csv')); pd.read_csv(R + 'three_way_detail.csv')"),
md("""Neither our model nor the bookmakers ever pick a draw, so three-way accuracy stays near 53-54%. This confirms the choice of the binary target.

## 9. Conclusions and limitations
- Logistic regression, refined using the VIF and residual diagnostics, matched the random forest in cross-validation, gave better-calibrated probabilities, and stayed interpretable.
- The biggest gains came from **better features** (Elo, 20-match shots), not from a more complex model. The baseline went from 61.6% to 66.2% in CV.
- Limitations: one test season (±5 pts sampling noise); no injury, line-up or transfer data (the bookmakers' remaining edge); `no_crowd` only corrects the past."""),
]
cells = [c for c in cells]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
path = ROOT / "Home_Win_Prediction.ipynb"
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, path)
print("executed ->", path)
