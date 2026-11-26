"""Stage 2 candidates - written only AFTER inspecting the develop-stage diagnostics
(reports/figures/03-06) and BEFORE the 2024-25 test season was evaluated.

What the diagnostics showed, and the response to each:
1. VIF (fig 03): gd20 = gf20 - ga20 exactly (VIF = inf); shot-share VIF > 100.
   Classic highest-VIF-first pruning then removed elo_diff.
2. Binned residuals vs elo_diff (fig 04) trend upward: the pruned model misses Elo signal.
   -> "target-aware" pruning: among VIF > 5 predictors drop the one least correlated
      with the training-fold outcome, so Elo survives.
3. rest_diff has p = 0.999 in the statsmodels fit -> compact model without it.
4. Max Cook's D = 0.004 (fig 04): no influential matches, nothing removed.
5. Train/validation gap (fig 06): deep trees overfit -> forests limited to the compact set
   with larger leaves.
"""
from sklearn.ensemble import RandomForestClassifier

from .features import ALL_FEATURES
from .models import logit, make

COMPACT = ["elo_diff", "shots20_diff", "shots_against20_diff", "venue_ppg_diff",
           "gf5_diff", "ga5_diff", "no_crowd"]


def refined_candidates():
    c = {}
    for C in (0.03, 0.1, 1.0):
        c[f"logit_vif_target_C{C}"] = ("logistic", make(logit(C), ALL_FEATURES, prune="target"))
        c[f"logit_compact_C{C}"] = ("logistic", make(logit(C), COMPACT))
    for leaf in (40, 80):
        c[f"forest_compact_d5_leaf{leaf}"] = ("forest", make(RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=leaf, max_features=0.5,
            random_state=0, n_jobs=-1), COMPACT))
    return c
