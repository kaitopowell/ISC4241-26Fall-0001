"""Render the measured experiment into a report and an executable review notebook."""
import base64
import html
import json
from pathlib import Path
import pandas as pd
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports'
lock=json.loads((R/'selection_lock.json').read_text())
cv=pd.read_csv(R/'cv_summary.csv').set_index('name')
test=pd.read_csv(R/'test_metrics.csv').set_index('name')
unc=json.loads((R/'test_uncertainty.json').read_text())
selected=lock['selected_model'];s=test.loc[selected];f=test.loc['forest_d8_leaf25'];prior=test.loc['prior']

def table(df):
    out=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        out.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(out)
comparison=pd.DataFrame({'Model':test.index,'CV log loss':cv.loc[test.index,'cv_log_loss'].values,
                         'Test log loss':test.log_loss.values,'Test accuracy':test.accuracy.values,
                         'Test macro F1':test.macro_f1.values,'Test draw recall':test.draw_recall.values})
sections=[]
def add(title,body,figures=()):sections.append((title,body,list(figures)))
add('Premier League prediction: final results',f'''Research question: can a random forest beat multinomial logistic regression on home/draw/away probabilities from pre-match team history?

The model selected before opening the test season was **{selected}**. Its held-out accuracy is **{s.accuracy:.1%}**, with log loss **{s.log_loss:.4f}**. Random forest test accuracy is **{f.accuracy:.1%}**, with log loss **{f.log_loss:.4f}**. Lower log loss is better. These are observed comparisons on one archived season, not proof of universal superiority.

There is no promise of a particular grade or live forecasting accuracy. The contribution is an auditable, leakage-controlled supervised-learning experiment.''')
add('Data and exact cleaning decisions','''The source is TheSportsDB data already saved in the team's original Premier League notebook. No API key or database is required. The saved retrieval timestamp is unknown. We extracted 1,520 scheduled events, excluded 161 unfinished events and 219 completed events from the incomplete 2024–25 season. The three complete seasons contain 1,140 matches.

Five prior recorded matches are required for both teams. This excludes 75 warm-up rows from modeling while retaining them in team histories: 695 development rows (2021–22/2022–23) and 370 held-out rows (2023–24). This means the results do not cover every promoted-team fixture. No mean imputation or outlier deletion was needed for these selected inputs. Optional API statistics were not used. Valid extreme scorelines remain; every exclusion and flagged scoreline is recorded in CSV.

The original notebook added the current result before calculating a last-five feature. Our replacement constructs every match's inputs before adding that day's results. A test changes all current/future scores and confirms earlier/current predictors stay unchanged.''',['01_class_counts.png','02_distributions.png'])
add('Features and collinearity','''We begin with home/away five-game points, goals for, goals against, goal difference, win fraction, capped rest days and prior Elo difference. Elo is a supplemental team-strength feature, not claimed as a taught model. Its constants were fixed and a no-Elo comparison is included.

The full heatmap shows points and wins are strongly correlated. More importantly, goal difference equals goals for minus goals against: full-model VIFs are infinite. Pairwise correlation alone would miss this exact multivariable dependency.

Refinement removes goal difference, uses home-minus-away differences, and iteratively removes predictors with VIF above 5 using only each training fold. The final logistic features are goals-for difference, goals-against difference, win-fraction difference, rest difference and Elo difference. Final VIFs range from about 1.00 to 3.51. This improves interpretability without claiming independent or causal predictors.''',['03_full_correlation.png','09_final_correlation.png'])
add('How validation stays honest','''Ten expanding chronological folds use strictly earlier dates for training and later dates for validation. Same-day fixtures stay together. The first 266 rows initialize training; 429 distinct later development fixtures receive out-of-fold predictions. Scaling and VIF screening are refitted inside each fold. Model parameters stay fixed within a validation block, while newly observed results may update later pre-match features, exactly as sequential forecasts would.

Nineteen configurations compare majority/prior baselines, initial and refined logistic regression, a class-weighted logistic model, a no-Elo logistic model, decision trees and random forests. The primary selection criterion is pooled out-of-fold multiclass log loss. C controls inverse L2 regularization; tree depth and leaf size control complexity. CV is also used for tuning, so it is not an unbiased nested-CV estimate. The untouched newest complete season supplies the final assessment.''',['06_cv_timeline.png','07_model_comparison.png','08_overfitting.png'])
add('Diagnostic evidence and refinement decisions',f'''Initial logistic OOF loss was {cv.loc['logistic_initial','cv_log_loss']:.4f}; refined logistic loss is {cv.loc[selected,'cv_log_loss']:.4f}. This combined change includes difference encoding, VIF screening and stronger regularization; do not attribute all improvement to VIF alone. Refined logistic and forest validation losses differ by only {cv.loc['forest_d8_leaf25','cv_log_loss']-cv.loc[selected,'cv_log_loss']:.4f}, so their ranking is close.

The confusion matrix reveals zero draw recall for the selected OOF model. Its draw probabilities can still be useful even when draw never has the largest probability. Class weighting increases OOF draw recall to {cv.loc['logistic_balanced','cv_draw_recall']:.1%}, but worsens log loss to {cv.loc['logistic_balanced','cv_log_loss']:.4f}. We keep the probability-focused winner and report the limitation. A different decision objective could justify a different choice, but must be specified before test evaluation.

Calibration and binned probability residuals reveal underprediction of some home-win probabilities and overprediction of some away-win probabilities. The extreme home-win bin has only two OOF examples: do not treat its shape as reliable. We do not fit an extra calibrator to these same OOF results and claim independent improvement. Monthly loss varies, showing that performance is not constant through time. No OLS normal-residual assumption is imposed on a categorical response.

At matched C=0.1, logistic OOF loss is {cv.loc['logistic_refined_C0.1','cv_log_loss']:.4f} with Elo versus {cv.loc['logistic_no_elo','cv_log_loss']:.4f} without it. This is evidence for added predictive information in this experiment, not a causal effect.''',['11_refined_oof_confusion.png','11_refined_oof_calibration.png','11_refined_oof_residuals.png','11_refined_oof_roc_pr.png'])
add('Held-out season: no model changes afterward',table(comparison)+f'''

The selected model's macro one-vs-rest AUC is {s.auc_macro_ovr:.4f}, multiclass Brier score is {s.brier_multiclass:.4f}, and draw recall is {s.draw_recall:.1%}. Accuracy should always be shown alongside these class-sensitive and probability metrics. The class-prior baseline accuracy is {prior.accuracy:.1%}.

Selected-minus-prior mean test log loss is {unc['estimate']:.4f}. The approximate paired week-block bootstrap 95% interval is [{unc['week_block_percentile_95'][0]:.4f}, {unc['week_block_percentile_95'][1]:.4f}]; negative values favor the selected model. Two thousand resamples preserve within-week blocks, but cross-week/team dependence may remain. This is conditional on the fitted models and does not establish a significant logistic-versus-forest difference.

The initial logistic model actually has lower test log loss (0.9311) than the selected refined model (0.9472). It also has slightly higher test accuracy. This is a useful example of validation rankings changing on new data; refinement did not improve every held-out metric. We retain the CV-selected model rather than choosing retrospectively.

The same preselected family representatives are reported for transparency. The selected model is not changed after inspecting this table. All results concern sequential forecasts within 2023–24 using a model fitted through 2022–23, not current 2026 fixtures.''',['20_test_confusion.png','20_test_calibration.png','20_test_roc_pr.png','20_test_residuals.png'])
add('Interpretation and limits','''The coefficient heatmap describes standardized inputs in multinomial logistic regression. To compare home versus away log odds, subtract their coefficients; an individual coefficient is not a percentage-point probability effect. The shallow decision tree illustrates nonlinear rules; its thresholds are on standardized values.

Repeated teams, one test season, stale historical data, promoted-team warm-up exclusions, narrow draw probabilities and absent lineup/injury/transfer information limit generalization. Valid surprises are not deleted. Further improvements require fresh independent seasons, not repeated tuning on this test season.

The original Flask pages and saved pickles are retained as legacy work, not the evaluated model. Use the new historical prediction command for the demonstration. It rejects unsupported future dates. See docs/METHODOLOGY.md, docs/COURSE_ALIGNMENT.md and docs/TEAM_PRESENTATION.md for details and rehearsal questions.''',['15_logistic_coefficients.png','16_tree.png'])
md=[];parts=[]
for title,body,figures in sections:
    md.extend(['## '+title,body]+[f'![{Path(f).stem}](figures/{f})' for f in figures])
    # Keep prose and tables readable without requiring an extra Markdown dependency.
    paragraphs=[]
    for p in body.split('\n\n'):
        if p.startswith('| '):paragraphs.append(comparison.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0))
        else:paragraphs.append('<p>'+html.escape(p).replace('**','')+'</p>')
    pics=''.join('<figure><img alt="'+html.escape(Path(f).stem)+'" src="data:image/png;base64,'+base64.b64encode((R/'figures'/f).read_bytes()).decode()+'"></figure>' for f in figures)
    parts.append('<section><h2>'+html.escape(title)+'</h2>'+''.join(paragraphs)+pics+'</section>')
(R/'RESULTS.md').write_text('\n\n'.join(md)+'\n')
(R/'final_report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Premier League — supervised learning project</title><style>body{margin:auto;max-width:1100px;padding:40px 24px;background:#f5f7fa;color:#152b40;font:18px/1.65 system-ui}section{background:white;padding:28px;margin-bottom:24px;border-radius:12px}h1,h2{line-height:1.2}img{width:100%;height:auto}figure{margin:24px 0}table{font-size:14px;width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}@media print{body{padding:0}section{break-inside:auto}}</style><h1>Predicting Premier League results</h1><p>ISC4241 · Reproducible final-project analysis · Archived 2021–2024 data</p>'''+''.join(parts)+'</html>')
nb=nbf.v4.new_notebook();cells=[nbf.v4.new_markdown_cell('# Premier League: class final project\n\nThis executed review notebook reads the saved experiment. Run the README commands to reproduce training and locked test evaluation. It never calls the original API or opens legacy pickles.'),nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nassert (ROOT/'reports/test_metrics.csv').exists(), 'Run from repository root after the analysis'\nR=ROOT/'reports'")]
for title,body,figures in sections:
    cells.append(nbf.v4.new_markdown_cell('## '+title+'\n\n'+body))
    if figures:cells.append(nbf.v4.new_code_cell('\n'.join(f"display(Image(filename=str(R/'figures/{f}')))" for f in figures)))
cells.append(nbf.v4.new_code_cell("display(pd.read_csv(R/'cleaning_audit.csv'))\ndisplay(pd.read_csv(R/'cv_splits.csv'))\ndisplay(pd.read_csv(R/'cv_summary.csv'))\ndisplay(pd.read_csv(R/'test_metrics.csv'))\ndisplay(pd.read_csv(R/'final_development_vif.csv'))"))
cells.append(nbf.v4.new_markdown_cell('## Historical matchup demonstration\nAn archived hypothetical matchup, not a live 2026 forecast. All history predates the selected date.'))
cells.append(nbf.v4.new_code_cell("from analysis.predict import predict_match\npredict_match('Arsenal','Chelsea','2024-05-20')"))
nb.cells=cells;nb.metadata.kernelspec={'display_name':'Python (ISC project)','language':'python','name':'isc-project'}
nbf.write(nb,ROOT/'Final_Project.ipynb')
print('Wrote report, standalone HTML and executable notebook')
