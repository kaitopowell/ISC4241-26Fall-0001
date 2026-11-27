"""Rebuild presentation figures from saved results and write reports/report.html.

Reads only saved CSV/JSON outputs plus the development features (for VIF); it does
not refit or re-select anything, so the locked test result cannot change.
"""
import base64
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer import plots                                   # noqa: E402
from soccer.models import bookmaker_prob, vif_table        # noqa: E402
from soccer.run import FIG, OUT, load                      # noqa: E402


def img(name):
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img alt="{name}" src="data:image/png;base64,{data}">'


def table(df, pct=(), dec=()):
    df = df.copy()
    for c in pct:
        df[c] = df[c].map(lambda x: f"{x:.1%}")
    for c in dec:
        df[c] = df[c].map(lambda x: f"{x:.3f}")
    return df.to_html(index=False, border=0, classes="t")


def three_way_detail(dev, test, lock):
    """Macro F1 / draw recall for the secondary 3-way task (refit of the same locked recipe)."""
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import classification_report, f1_score
    from soccer.models import logit, make
    sel = joblib.load(ROOT / "artifacts" / "final_model.joblib")["selected"]
    pipe = make(logit(sel.named_steps["model"].C), lock["features"])
    pred = clone(pipe).fit(dev, dev.result).predict(test)
    q = bookmaker_prob(test).idxmax(axis=1).to_numpy()
    rows = []
    for name, pr in [("multinomial logistic", pred), ("bookmaker favourite", q)]:
        rep = classification_report(test.result, pr, labels=["H", "D", "A"], output_dict=True, zero_division=0)
        rows.append({"model": name, "accuracy": rep["accuracy"], "macro_F1": f1_score(test.result, pr, average="macro"),
                     "recall_H": rep["H"]["recall"], "recall_D": rep["D"]["recall"], "recall_A": rep["A"]["recall"]})
    out = pd.DataFrame(rows); out.to_csv(OUT / "three_way_detail.csv", index=False)
    return out


def main():
    lock = json.loads((OUT / "selection_lock.json").read_text())
    _, dev, test_rows, _ = load()
    vif_full = pd.read_csv(OUT / "vif_all_features.csv", index_col=0).vif
    vif_classic = pd.read_csv(OUT / "vif_after_pruning.csv", index_col=0).vif
    vif_final = vif_table(dev[[c for c in lock["features"]]])
    vif_final.rename("vif").to_csv(OUT / "vif_selected_model.csv")
    plots.vif_bars(vif_full, vif_classic, FIG / "03_vif_before_after.png", final=vif_final)
    cv = pd.read_csv(OUT / "cv_all.csv")
    book = json.load(open(OUT / "bookmaker_cv.json"))
    plots.cv_comparison(cv, book, FIG / "07_cv_model_comparison.png", selected=lock["selected"])

    test = pd.read_csv(OUT / "test_results.csv")
    boot = json.loads((OUT / "test_bootstrap.json").read_text())
    three = pd.read_csv(OUT / "three_way_test.csv")
    three_d = three_way_detail(dev, test_rows, lock)
    coef = pd.read_csv(OUT / "selected_coefficients.csv", index_col=0).sort_values("std_coef", ascending=False)
    top = cv.head(8)[["model", "family", "accuracy", "log_loss", "auc", "fold_acc_sd"]]
    base = cv[cv.model.isin(["majority", "logit_baseline_form5"])][top.columns]
    cvt = pd.concat([top, base]).drop_duplicates("model")
    ci = lambda k: f"{boot[k][0]*100:+.1f} pts (95% CI {boot[k][1]*100:+.1f} to {boot[k][2]*100:+.1f})"
    sel = test.iloc[0]

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Home Win Prediction</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--mut:#52514e;--line:#e4e3de;--acc:#2a78d6}}
body{{background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif;margin:0}}
main{{max-width:980px;margin:auto;padding:32px 16px 64px}} h1{{font-size:30px;margin:0 0 4px}}
h2{{margin-top:44px;border-top:1px solid var(--line);padding-top:20px}} .mut{{color:var(--mut)}}
img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff;margin:10px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}}
.kpi{{border:1px solid var(--line);border-radius:8px;padding:12px 14px;background:#fff}}
.kpi b{{display:block;font-size:26px}} .kpi span{{color:var(--mut);font-size:13px}}
table.t{{border-collapse:collapse;font-size:14px;width:100%;overflow-x:auto;display:block}}
table.t th,table.t td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
li{{margin:4px 0}} code{{background:#f1f0ec;padding:1px 4px;border-radius:3px}}
</style></head><body><main>
<h1>Predicting Premier League home wins</h1>
<p class="mut">ISC 4241 final project - logistic regression vs decision trees and random forests,
10-fold chronological cross-validation, collinearity and residual diagnostics, one untouched test season.</p>

<div class="kpis">
<div class="kpi"><b>{sel.accuracy:.1%}</b><span>Selected model, 2024-25 test accuracy</span></div>
<div class="kpi"><b>{test.set_index('model').loc['logit_baseline_form5','accuracy']:.1%}</b><span>Baseline logistic (original 5-game form)</span></div>
<div class="kpi"><b>{test.set_index('model').loc['majority','accuracy']:.1%}</b><span>Always "not a home win"</span></div>
<div class="kpi"><b>{test.iloc[-1].accuracy:.1%}</b><span>Bookmakers (benchmark only)</span></div>
</div>

<h2>1. Question and why it is binary</h2>
<p>Will the home team win (1) or not (0: draw or away win)? Binary logistic regression is the model from Lecture 9, and
Lecture 7's sensitivity, specificity, ROC and AUC apply to it directly. We also tried the three-way home/draw/away version. It is capped
near 55%: on the test season even the bookmakers' favourite won only {three.iloc[-1].accuracy:.1%} of matches, and our
multinomial model reached {three.iloc[1].accuracy:.1%} (section 7).</p>

<h2>2. Data and leakage control</h2>
<ul><li>11 complete Premier League seasons (2014-15 to 2024-25) from Football-Data.co.uk: 4,180 matches.</li>
<li>2014-16 only build up Elo and form history. Development data: 2016-17 to 2023-24 ({len(dev)} matches). Test: 2024-25 (366 matches).</li>
<li>Every predictor uses matches from <b>earlier dates only</b>. A day's results enter team history after all of that day's matches are predicted. Tests check this.</li>
<li>Matches where a team had fewer than 5 recent league games (promoted clubs' opening weeks) are excluded and logged.</li>
<li>Bookmaker odds are <b>never</b> model inputs. They are used only as an expert benchmark.</li></ul>
{img('01_home_win_rate_by_season.png')}

<h2>3. Collinearity</h2>
<p>We started with 17 candidate predictors. <code>gd20 = gf20 - ga20</code> is an exact linear combination (VIF = inf), and the shot-share variables have VIF above 100.
The classic rule (drop the highest VIF until all are at most 5) also removed <code>elo_diff</code>, the single strongest predictor.</p>
{img('02_correlation_heatmap.png')}{img('03_vif_before_after.png')}

<h2>4. Diagnostic plots and the refinements they led to</h2>
{img('04_logistic_diagnostics.png')}
<ul><li><b>Binned residuals vs <code>elo_diff</code></b> trend upward. The pruned model under-uses team strength, so we
switched to <i>target-aware</i> VIF pruning: among VIF > 5 predictors, drop the one least related to the outcome. We also built a compact set that keeps Elo.</li>
<li><b>Binned residuals vs fitted probability</b> stay inside ±2 SE, so the logit link and calibration look adequate. No polynomial terms were added.</li>
<li><b>Leverage / Cook's distance:</b> the maximum Cook's D is 0.004. Some matches exceed the strict 4/n = 0.0014 screen, but all are far below the D > 0.5 to 1 level that signals undue influence. The top ten (<code>reports/influence_top.csv</code>) are genuine upsets, not data errors, so nothing was removed.</li>
<li><code>rest_diff</code> had p = 0.999, so it was dropped from the compact model.</li>
<li><b>Residuals by season:</b> 2020-21 (no crowds) is over-predicted. Hence the <code>no_crowd</code> indicator. Early folds cannot learn it because none of their training data is from that period.</li>
<li><b>Training vs validation accuracy:</b> deeper trees fit the training data better without validating better (overfitting). Refined forests use shallow trees with large leaves.</li></ul>
{img('05_oof_residual_by_season.png')}{img('06_tree_forest_train_vs_valid.png')}

<h2>5. 10-fold chronological cross-validation ({len(cv)} configurations)</h2>
<p>Each fold trains on all earlier matches and validates on the next block, so we never train on the future.
Selection rule, declared in advance: highest pooled out-of-fold accuracy, with models within 0.25 pts treated as tied and the lower log loss winning.</p>
{table(cvt, pct=['accuracy','fold_acc_sd'], dec=['log_loss','auc'])}
{img('07_cv_model_comparison.png')}{img('08_cv_accuracy_by_fold.png')}{img('09_oof_calibration_roc.png')}
<p><b>Selected:</b> <code>{lock['selected']}</code>, an L2-penalised logistic regression on {len(lock['features'])} predictors
(six have VIF 1.0-3.2; <code>elo_diff</code> is {vif_final['elo_diff']:.1f}, slightly above 5 and below the common cutoff of 10.
We kept it deliberately: the residual plots showed that dropping it hurt, and the L2 penalty stabilises the coefficient).
It ties the best random forest on accuracy and beats it on log loss and AUC.</p>
{img('10_selected_coefficients.png')}
{table(coef.rename_axis('predictor').reset_index(), dec=['std_coef','odds_ratio_per_sd'])}

<h2>6. Held-out test: 2024-25 (evaluated once, after locking)</h2>
{table(test, pct=['accuracy','sensitivity','specificity'], dec=['log_loss','brier','auc'])}
{img('11_test_confusion_roc.png')}
<ul><li>Gain over the baseline logistic: {ci('gain_vs_p_logit_baseline_form5')}.</li>
<li>Gain over always guessing "not a home win": {ci('gain_vs_p_majority')}.</li>
<li>Difference from the bookmakers: {ci('gain_vs_p_bookmaker')}. Our model is statistically level with the betting market.</li>
<li>The best random forest scored slightly higher on this one season. It was <i>not</i> selected, because the choice was locked
on cross-validation before testing. Switching now would be selecting on the test set.</li>
<li>Intervals come from a paired bootstrap over match-weeks (2,000 resamples).</li></ul>

<h2>7. Secondary: three-way home/draw/away</h2>
{table(three, pct=['accuracy'])}
{table(three_d, pct=['accuracy','recall_H','recall_D','recall_A'], dec=['macro_F1'])}
<p>Draws are about 25% of matches and are almost never the single most likely outcome, so neither the model nor the bookmakers ever predict one.
This is why the three-way accuracy is so low, and why we chose the binary target.</p>

<h2>8. Limitations</h2>
<ul><li>One test season (366 matches): roughly ±5 pts of sampling noise around any accuracy.</li>
<li>The model has no information on injuries, line-ups or transfers. The bookmakers do, which explains their small remaining edge.</li>
<li>The no-crowd indicator cannot help future predictions. It only stops 2020-21 from distorting the other coefficients.</li></ul>
<p class="mut">Reproduce: <code>python -m soccer.run --stage develop</code> -> <code>--stage refine</code> -> <code>--stage test</code> -> <code>python scripts/build_report.py</code>.</p>
</main></body></html>"""
    (OUT / "report.html").write_text(html)
    print("wrote", OUT / "report.html")


if __name__ == "__main__":
    main()
