"""Diagnostics and portable report for the accuracy-focused follow-up."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse
import base64
import json
import html
import numpy as np
import pandas as pd
import joblib
import nbformat
from analysis import plots
from analysis.models import CLASSES,vif_table
from sklearn.metrics import classification_report
from experiments.improve import OUT,DATA,ART,SHOTS,fingerprint

parser=argparse.ArgumentParser();parser.add_argument('--development',action='store_true');args=parser.parse_args()
lock=json.loads((OUT/'selection_lock.json').read_text());assert lock['fingerprint']==fingerprint()
selected=lock['selected'];cv=pd.read_csv(OUT/'cv_summary.csv').set_index('name')
f=pd.read_csv(DATA/'features.csv',parse_dates=['date']);dev=f[(f.season>='2016-2017')&(f.season<'2024-2025')]
plots.FIG=OUT/'figures';plots.FIG.mkdir(exist_ok=True)
plots.heatmap(dev[SHOTS],'full_correlation','Expanded candidate inputs: development only')
vif_table(dev[SHOTS].fillna(dev[SHOTS].median())).to_csv(OUT/'full_vif.csv',index=False)
vif_table(dev[lock['features']].fillna(dev[lock['features']].median())).to_csv(OUT/'selected_vif.csv',index=False)
for name,prefix in [(selected,'selected_oof'),('original_recipe','recipe_oof')]:
    oof=pd.read_csv(OUT/f'oof_{name}.csv',parse_dates=['date'])
    plots.probability_diagnostics(oof,oof[['p_A','p_D','p_H']].to_numpy(),prefix,'Development OOF: '+name)
import matplotlib.pyplot as plt
best=cv.sort_values('accuracy').tail(10)
plt.figure(figsize=(10,6));plt.barh(best.index,best.accuracy);plt.xlabel('Pooled out-of-fold accuracy');plt.title('Top development configurations');plots.save('comparison')
folds=pd.read_csv(OUT/'fold_metrics.csv');subset=folds[folds.name.isin([selected,'original_recipe'])]
fig,axes=plt.subplots(1,2,figsize=(11,4))
for name,g in subset.groupby('name'):
    axes[0].plot(g.fold,g.accuracy,'o-',label=name)
    axes[1].plot(g.fold,g.train_accuracy-g.accuracy,'o-',label=name)
axes[0].set(title='Validation accuracy by chronological fold',xlabel='Fold',ylabel='Accuracy');axes[1].set(title='Training minus validation accuracy',xlabel='Fold',ylabel='Gap');axes[0].legend(fontsize=7);axes[1].axhline(0,color='gray',ls='--');plots.save('fold_diagnostics')
if args.development:
    print('Development diagnostic figures and VIF tables saved.');sys.exit()

test=pd.read_csv(OUT/'test_metrics.csv').set_index('name');s=test.loc[selected];old=test.loc['original_recipe'];prior=test.loc['prior']
pred=pd.read_csv(OUT/f'test_{selected}.csv',parse_dates=['date'])
ref=pd.read_csv(OUT/'test_original_recipe.csv',parse_dates=['date']);assert pred.match_id.equals(ref.match_id)
p=pred[['p_A','p_D','p_H']].to_numpy();plots.probability_diagnostics(pred,p,'fresh_test','New 2024–25 evaluation: '+selected)
pd.DataFrame(classification_report(pred.outcome,pred.predicted,labels=CLASSES,zero_division=0,output_dict=True)).T.to_csv(OUT/'classification_report.csv')
# Independent verification and paired uncertainty; no additional models selected here.
assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
assert np.isclose((pred.outcome==pred.predicted).mean(),s.accuracy)
idx=np.array([list(CLASSES).index(y) for y in pred.outcome]);assert np.isclose(-np.log(p[np.arange(len(p)),idx]).mean(),s.log_loss)
delta=(pred.predicted==pred.outcome).astype(int)-(ref.predicted==ref.outcome).astype(int)
weeks=pred.date.dt.to_period('W');blocks=[delta[weeks==w].to_numpy() for w in weeks.unique()]
rng=np.random.default_rng(42);samples=[np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]).mean() for _ in range(2000)]
ci=np.quantile(samples,[.025,.975]);(OUT/'accuracy_uncertainty.json').write_text(json.dumps({'selected_minus_original_recipe':float(delta.mean()),'paired_week_block_95':ci.tolist(),'resamples':2000,'note':'Conditional approximate interval; between-week/team dependence and model selection uncertainty not fully represented.'},indent=2))
comparison=test[['accuracy','macro_f1','draw_recall','log_loss','auc_macro_ovr']].reset_index()
def mdtable(d):
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
sections=[('What changed',f'''We tested a genuine improvement attempt after the original 56.5% accuracy result. The new experiment uses 4,180 fixtures from eleven complete seasons from Football-Data.co.uk (2014–15 through 2024–25). Two seasons initialize history; {lock['development_n']:,} eligible fixtures from 2016–24 train and validate models. One completely blank row in the 2014–15 source is dropped. Optional shot values are imputed within folds if missing. Valid extreme results remain.

The old 2023–24 test is now explicitly development data. The new evaluation is 2024–25, selected before its prediction scores were computed. Earlier results within an evaluation block update later pre-match histories; model parameters remain fixed. No betting odds or current-match shots are used. The original reports and draw-focused follow-up remain intact.''',[]),
('Development evidence and selected model',f'''Ten expanding chronological folds compare {lock['candidates']} configurations. Accuracy is now the primary objective, with log loss breaking ties. This was changed explicitly following the user's request, not secretly after seeing test rankings. The selected model is {selected}, with OOF accuracy {cv.loc[selected,'accuracy']:.2%}, compared with {cv.loc['original_recipe','accuracy']:.2%} for the old recipe refitted on the same development rows. Candidate models include multinomial logistic regression, decision trees and random forests; complexity does not guarantee higher accuracy.

New features use 20-match goals, points and shot rates, plus ten venue-specific games, Elo and rest. All statistics are lagged by date. Tests perturb current and future scores/shots and verify past/current inputs do not change. The older five-game recipe is a fair same-data comparator, not the exact frozen 2023–24 model.''',['comparison.png','fold_diagnostics.png']),
('Collinearity and diagnostic refinement',f'''The full eleven-input design has redundant information, with VIFs up to 14.44. Before opening the new test, four additional logistic configurations applied training-only VIF pruning after median imputation. They are included in the 42-model comparison; selection remains based on OOF accuracy. The selected inputs are: {', '.join(lock['features'])}.

The selected pruned model retains seven predictors with final VIFs from 1.01 to 4.90, all below the declared threshold of 5. Its OOF accuracy is 55.34%, compared with 55.11% for the unpruned shot model. This is a small observed improvement, not proof that VIF pruning always helps. L2 regularization and remaining correlation still limit individual coefficient interpretation. Calibration, confusion and train–validation gaps are shown, not used to tune against the new test.''',['full_correlation.png','selected_oof_confusion.png','selected_oof_calibration.png']),
('New-season evaluation',mdtable(comparison)+f'''

On {len(pred)} identical eligible 2024–25 fixtures, the selected model scores {s.accuracy:.2%} accuracy; the refitted original recipe scores {old.accuracy:.2%}; the prior baseline scores {prior.accuracy:.2%}. The paired difference versus the old recipe is {(s.accuracy-old.accuracy)*100:+.2f} percentage points. Its approximate week-block 95% interval is [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] points. An interval containing zero does not establish a reliable improvement.

The original 56.5% was measured on 2023–24. Comparing that number with this season's accuracy is not a controlled improvement estimate. Use the same-season comparator above. We report the result whether better or worse and do not choose another model after seeing it.''',['fresh_test_confusion.png','fresh_test_calibration.png','fresh_test_roc_pr.png','fresh_test_residuals.png']),
('What to tell the professor',f'''We expanded training history, tested richer pre-match information, compared 42 configurations with chronological CV, checked redundancy and locked selection before evaluating a new season. The selected model's new-season draw recall is {s.draw_recall:.1%}, so accuracy alone still hides an important weakness. The separate draw-refinement experiment explores a different objective; do not mix its metrics with this accuracy experiment.

Limits: match outcomes remain uncertain; there are no lineups, injuries or player-strength variables. Relegated teams can carry stale history from earlier Premier League seasons. Five-game warm-up still excludes newly observed teams: evaluate only eligible fixtures and state the denominator. A larger search can overfit validation; the fresh season is the assessment. These results do not support promising substantially higher accuracy. A change of target (such as excluding draws) would be a different task, not an improvement on this one.

Source: https://www.football-data.co.uk/englandm.php. File URLs and checksums are in data/expanded/provenance.json. See experiments/PROTOCOL.md for the prespecified plan and development-only refinement.''',[])]
md=[];parts=[];cells=[nbformat.v4.new_markdown_cell('# Accuracy improvement experiment\n\nThe original evaluation is preserved. This notebook reviews actual expanded-data results.'),nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nR=Path.cwd()/'reports/expanded'\nassert R.exists()")]
for title,text,figs in sections:
    md.extend(['## '+title,text]+[f'![{Path(f).stem}](figures/{f})' for f in figs])
    paras=[]
    for paragraph in text.split('\n\n'):
        paras.append(comparison.to_html(index=False,float_format=lambda v:f'{v:.4f}') if paragraph.startswith('| ') else '<p>'+html.escape(paragraph)+'</p>')
    images=''.join('<img alt="'+html.escape(Path(f).stem)+'" src="data:image/png;base64,'+base64.b64encode((plots.FIG/f).read_bytes()).decode()+'">' for f in figs)
    parts.append('<section><h2>'+html.escape(title)+'</h2>'+''.join(paras)+images+'</section>')
    cells.append(nbformat.v4.new_markdown_cell('## '+title+'\n\n'+text))
    if figs:cells.append(nbformat.v4.new_code_cell('\n'.join(f"display(Image(filename=str(R/'figures/{f}')))" for f in figs)))
cells.append(nbformat.v4.new_code_cell("display(pd.read_csv(R/'cv_summary.csv'))\ndisplay(pd.read_csv(R/'selected_vif.csv'))\ndisplay(pd.read_csv(R/'test_metrics.csv'))\ndisplay(pd.read_csv(R/'classification_report.csv'))"))
(OUT/'RESULTS.md').write_text('\n\n'.join(md)+'\n')
(OUT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Accuracy improvement experiment</title><style>body{max-width:1100px;margin:auto;padding:30px;background:#f3f6fa;color:#203244;font:18px/1.6 system-ui}section{background:white;padding:25px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}table{font-size:14px;width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc}</style><h1>Can we improve prediction accuracy?</h1>'+''.join(parts)+'</html>')
nb=nbformat.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'name':'isc-project','display_name':'Python (ISC project)','language':'python'};nbformat.write(nb,ROOT/'Accuracy_Improvement.ipynb')
print(comparison.to_string(index=False));print('Paired accuracy difference CI:',ci);print('Wrote expanded report and notebook.')
