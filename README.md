# Premier League results: a class-aligned supervised-learning project

**Question:** Can a random forest predict home/draw/away outcomes better than multinomial logistic regression using only information available before a match?

Start with [the measured results](reports/RESULTS.md), [the executed notebook](Final_Project.ipynb), or download/open [the standalone HTML report](reports/final_report.html). The HTML contains its figures and works offline. This branch replaces the assessed analysis with a reproducible historical experiment; the original notebooks and Flask site remain as legacy material.

## Accuracy improvement follow-up

After the request to improve 56.5% accuracy, we expanded to eleven seasons and evaluated 42 configurations with longer form and historical shots. On a **new 2024–25 season**, the selected model achieved **53.9%**, versus **52.5%** for the original recipe refitted on the same data and **41.1%** for the prior baseline. The paired gain is small and its uncertainty interval includes zero. The earlier 56.5% was on a different season; this is not evidence of a large improvement.

Read [the follow-up findings](reports/expanded/RESULTS.md), [the executed follow-up notebook](Accuracy_Improvement.ipynb), or [the reproduction guide](experiments/README.md). The original experiment and existing draw-refinement analysis are preserved. The follow-up changes the primary objective to accuracy explicitly and uses a fresh evaluation season after the original test had already been seen.

## What is included

- Audited extraction of three complete TheSportsDB seasons already cached in the original notebook; no API credentials or database needed.
- Pre-match five-game form and Elo, with tests against current-result, future-result and same-day leakage.
- Majority and class-prior baselines; initial/refined multinomial logistic regression; tuned decision trees and random forests; class-weight and no-Elo comparisons (19 configurations).
- Ten expanding chronological folds, train-only scaling/VIF selection and a held-out 2023–24 season.
- Full correlation heatmap, before/after VIF, class distributions, train/validation gaps, calibration, probability residuals, ROC/PR, confusion matrices, error examples and approximate block-bootstrap uncertainty.
- A historical matchup demonstration, course alignment notes, and a timed team presentation plan with Kiro owning the technical section.

## Reproduce from repository root

Python 3.9.6 was used; exact package versions are pinned. Python 3.10–3.12 may work but were not tested. Allow several minutes for the model comparisons and figures.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pytest -q
python -m analysis.run --stage develop
# Inspect development diagnostics and the frozen selection before first test evaluation.
python -m analysis.run --stage test
python scripts/build_deliverables.py
python scripts/verify_results.py
```

Committed reports are the completed experiment. A fresh clone needs the development command to recreate its local fitted bundle (excluded from Git). Re-evaluation of the exact same experiment checks that held-out metrics match the saved values. A code/data fingerprint prevents overwriting the test results with a changed experiment. **Do not delete the test manifest to tune against this season.** New modeling ideas require a new untouched evaluation period. Reproducing existing results is not a new independent test.

To execute the review notebook after regenerating it:

```bash
python -m ipykernel install --prefix .venv --name isc-project --display-name 'Python (ISC project)'
JUPYTER_PATH="$PWD/.venv/share/jupyter" python scripts/execute_notebook.py
```

Historical prediction example:

```bash
python -m analysis.predict --home Arsenal --away Chelsea --date 2024-05-20
```

This is a hypothetical archived matchup. It does not forecast current fixtures; dates beyond the snapshot's supported horizon are rejected. The fitted model stays trained through 2022–23, while past results update the pre-match history.

## Read and present

| File | Purpose |
|---|---|
| [Final_Project.ipynb](Final_Project.ipynb) | Executed walkthrough, figures, tables and example |
| [reports/RESULTS.md](reports/RESULTS.md) | Actual findings and diagnostic decisions |
| [reports/final_report.html](reports/final_report.html) | Portable report with embedded figures |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Cleaning, assumptions, leakage, metrics and limitations |
| [docs/COURSE_ALIGNMENT.md](docs/COURSE_ALIGNMENT.md) | Connections to the available lecture materials |
| [docs/TEAM_PRESENTATION.md](docs/TEAM_PRESENTATION.md) | Roles, 11-minute outline and questions to rehearse |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Executed checks and numerical warning investigation |
| [docs/FEATURE_GUIDE.md](docs/FEATURE_GUIDE.md) | Definitions, equations and worked example |
| [reports/selection_lock.json](reports/selection_lock.json) | CV choice, cutoffs, versions and fingerprint |
| [data/raw/provenance.json](data/raw/provenance.json) | Source notebook checksum and extraction cells |
| reports/*.csv | Row-level audit, fold results, VIF and saved predictions |
| analysis/ | Extraction, feature engineering, validation, plots and prediction |
| tests/ | Leakage, chronology, class order, data-integrity and numerical checks |

The dataset is an archived snapshot, not live 2026 data. The newest **complete cached season** is 2023–24. Check source reuse terms before redistributing independently; this repository's saved source does not establish a new data license. The old `proj4flask.py`, database dependencies and pickle files are not part of the evaluated pipeline. Do not present their old accuracy as a comparable result.

## Follow-up: recognizing draws (five-member team)

Read [the new results](reports/draw_refinement/RESULTS.md), [the follow-up notebook](Draw_Recognition_Followup.ipynb), or open [the offline follow-up report](reports/draw_refinement/report.html). This preserves the original experiment and uses **10 outer × 3 inner chronological folds** to tune models and draw decision rules. The already-opened 2023–24 season is now development data; the later partial 2024–25 snapshot supplies a separate final check.

The draw-aware forest caught 10 of 57 draws and reached 51.4% accuracy versus a same-row reference of 50.0%. However, nested validation lost accuracy, no configuration met both declared development goals, and the later accuracy-difference interval includes zero. It is a draw-sensitive alternative, **not a demonstrated accuracy upgrade**. See the report for the class-metric trade-off and partial-season limitations. The presentation plan now assigns work to five members, with Kiro explaining the main technical section.

The follow-up protocol is in [PLAN.md](experiments/draw_refinement/PLAN.md). To verify reproducibility without overwriting either frozen experiment:

```bash
source .venv/bin/activate
python -m pytest -q
python scripts/reproduce_draw.py
```

For a genuinely new experiment with a new untouched holdout, the module has separate `--stage develop` and `--stage test` commands. Its manifest deliberately refuses re-development after evaluation. Do not remove that protection to fish for a better result on the same holdout. The reproduction script instead runs the identical frozen procedure in a disposable directory and compares its saved outputs.
