# Accuracy-focused follow-up protocol

This follow-up is requested after the original test accuracy (56.5%) was seen. It does not erase the original experiment or relabel that season as untouched.

Before evaluating the new test season:
- Source: Football-Data.co.uk Premier League seasons 2014–15 through 2024–25, with dates, results and match statistics.
- 2014–16 provide feature warm-up; eligible 2016–24 fixtures are development data, including the previously seen 2023–24 season.
- 2024–25 is the new evaluation season, selected before computing candidate test scores. Do not search for a better-looking test season.
- Compare the original five-match recipe (refit on expanded development), prior, logistic, decision tree and random forest models.
- Candidate new features: 20-match scoring/points/shot rates, ten venue-specific games, Elo and rest. Current-match shots are prohibited; only earlier dates enter history. Optional missing values are imputed within training folds.
- Ten expanding chronological folds. Select highest pooled out-of-fold accuracy, break ties by lower log loss. This objective change is explicit; retain macro F1, draw recall, log loss and calibration to expose trade-offs.
- Evaluate only the selected model, old-recipe comparator and prior on the new season; report all three. Compare on identical eligible fixtures.
- More data and feature changes occur together. Do not attribute improvements to a single factor without an ablation. Do not compare different-season accuracies as if they were a paired improvement.

Development-only refinement: the eleven-input shot model had VIFs up to 14.44. Before test evaluation, add four logistic C values with median imputation followed by fold-local VIF pruning to <=5. This raises the candidate count from 38 to 42; no held-out scores informed this change.
