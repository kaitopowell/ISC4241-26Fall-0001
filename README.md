# Predicting Premier League Home Wins

UCF Data Science I · ISC 4241 · Fall 2026 - final project (clean redo, branch `home-win-prediction`)

**Question:** Using only information available before kickoff, can we predict whether the home team wins? Does a refined logistic regression do as well as decision trees and random forests?

| 2024-25 held-out season (366 matches) | Accuracy | AUC |
|---|---:|---:|
| Always "not a home win" (majority) | 58.7% | 0.50 |
| Baseline logistic: original 5-game form features | 62.6% | 0.63 |
| **Selected: refined logistic regression (7 predictors, L2)** | **65.6%** | **0.71** |
| Best random forest (comparator, not selected) | 66.9% | 0.71 |
| Bookmakers (benchmark only, never a model input) | 66.4% | 0.73 |

In 10-fold chronological CV (2016-17 to 2023-24), the selected model scored 66.2%, against 61.6% for the baseline logistic and 55.3% for the majority guess. It tied the best forest (66.4%) and had the best log loss and AUC of all 31 configurations.

**Start here:** [Home_Win_Prediction.ipynb](Home_Win_Prediction.ipynb) (executed) · [reports/report.html](reports/report.html) (offline report with every figure) · [docs/RESULTS.md](docs/RESULTS.md) (findings and presentation talking points)

## What the project does

1. **Data** - 11 complete Premier League seasons (2014-15 to 2024-25, 4,180 matches) from Football-Data.co.uk, validated season by season ([provenance](data/football_data/provenance.json)).
2. **Leak-free features** - Elo rating, 5- and 20-match form, shots and shots on target, venue-specific form, rest days, and a no-crowd (COVID) indicator. All are computed from earlier dates only, and tests prove it.
3. **Models** - majority baseline; **baseline logistic regression** (the original project's 5-game form); full, VIF-pruned, target-aware VIF-pruned, lasso (L1) and compact logistic models; decision trees; random forests.
4. **10-fold chronological cross-validation** - expanding window, date-aligned. Pruning and scaling are refit inside every fold.
5. **Collinearity** - correlation heatmap and VIF. `gd20 = gf20 - ga20` is exactly collinear (VIF = inf), and classic VIF pruning wrongly dropped Elo.
6. **Diagnostic plots, then refinement** - binned residuals (vs fitted and vs predictors), leverage and Cook's distance, residuals by season, train vs validation gap. Each finding led to a specific change; see `soccer/refine.py`.
7. **One test** - the choice was locked (with a code and data fingerprint) before 2024-25 was scored. There is a paired week-block bootstrap CI for every comparison.

The team brief's three-way home/draw/away version is reported as a secondary result: 52.7% accuracy, macro F1 0.39, against the bookmakers' 54.4%. Draws are never the most likely outcome, so draw recall is 0 for us and for the bookmakers. That ceiling is why the binary target is the main analysis.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m soccer.run --stage develop    # initial grid + diagnostics
python -m soccer.run --stage refine     # diagnostic-driven candidates, locks the choice
python -m soccer.run --stage test       # single evaluation on 2024-25
python scripts/build_report.py          # figures + reports/report.html
python scripts/make_notebook.py         # executes Home_Win_Prediction.ipynb
```

`--stage test` refuses to run if any modelling code or data changed after the lock.

## Layout

| Path | Contents |
|---|---|
| `soccer/data.py` | load and validate the CSVs |
| `soccer/features.py` | leak-free Elo and rolling features |
| `soccer/models.py` | candidates, VIF, chronological folds, metrics |
| `soccer/refine.py` | refinements, each justified by a diagnostic |
| `soccer/run.py` | develop, refine and test stages, bootstrap, three-way task |
| `soccer/plots.py` | all figures |
| `reports/` | CV tables, VIF tables, influence, lock, test results, figures, HTML report |
| `tests/test_pipeline.py` | leakage (current, future and same-day), chronology, VIF, odds exclusion |

Legacy files from the original project (Flask app, LIGA MX and BESTANOTHER notebooks, `static/`, `templates/`) are untouched. They are not part of the evaluated pipeline.
