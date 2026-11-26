# Follow-up experiments

The original `analysis/` pipeline and its frozen 2023–24 evaluation remain unchanged.

## Accuracy-focused expanded-data experiment

Read `reports/expanded/RESULTS.md` or open `reports/expanded/report.html` after the run. `Accuracy_Improvement.ipynb` is the executed review notebook. This is a follow-up to the user's request for higher accuracy, with a new 2024–25 evaluation period. It does not turn the already-seen original test back into untouched data.

```bash
.venv/bin/python -m experiments.improve --stage develop
.venv/bin/python scripts/report_expanded.py --development
# Inspect OOF diagnostics; the chosen model is frozen before new test scores.
.venv/bin/python -m experiments.improve --stage test
.venv/bin/python scripts/report_expanded.py
JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python scripts/execute_notebook.py Accuracy_Improvement.ipynb
.venv/bin/python -m pytest -q
```

Data and checksums are included in `data/expanded/`. Source: https://www.football-data.co.uk/englandm.php. Raw files contain additional fields including betting odds, but the loader explicitly selects only fixture details, final outcomes and shot statistics. Current fixture outcomes and shots are **never** model predictors; they update only future dates' histories. Do not replace that allowlist with all numeric columns.

The model uses all eligible fixtures, including draws. Accuracy is primary in this follow-up, while log loss, macro F1 and draw recall remain visible. Ten chronological folds compare the old recipe and new models on the same rows. Each fit learns its own scaling, imputation and (when applicable) VIF removal. A code/data fingerprint prevents overwriting test results with changed code. Re-running an identical evaluated configuration verifies saved metrics; changing the experiment requires a new untouched evaluation period.

The source has 4,180 fixtures across eleven complete seasons. Development has 2,985 eligible fixtures from 2016–24, after two warm-up seasons. A team still needs five prior recorded matches; exclusions are saved by match and season. Twenty-game and venue histories use available past records when fewer than twenty/ten exist. They can span relegation gaps, a documented limitation. Unusual legitimate scorelines are retained.

## Draw-focused experiment

`experiments/draw_refinement/` and its reports are a separate existing experiment targeting draw recognition. Preserve its protocol and conclusions. Higher accuracy and better minority-class recall are different objectives; neither experiment's results should be substituted for the other's test results.
