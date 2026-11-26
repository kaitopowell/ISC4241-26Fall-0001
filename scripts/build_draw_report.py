"""Publish follow-up findings without changing either frozen experiment."""
from pathlib import Path
import base64,html,json
import numpy as np
import pandas as pd
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/draw_refinement'
sns.set_theme(style='whitegrid');F=R/'figures';F.mkdir(exist_ok=True)
cv=pd.read_csv(R/'nested_summary.csv').set_index('approach')
test=pd.read_csv(R/'test_metrics.csv').set_index('approach')
pred=pd.read_csv(R/'test_predictions.csv');search=pd.read_csv(R/'final_development_search.csv')
lock=json.loads((R/'selection_lock.json').read_text());unc=json.loads((R/'uncertainty.json').read_text())

def save(name):plt.tight_layout();plt.savefig(F/name,dpi=160,bbox_inches='tight');plt.close()
fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,metric in zip(axs,['accuracy','macro_f1','draw_recall']):
    data=pd.DataFrame({'Nested CV reference':[cv.loc['reference',metric]],'Nested CV draw-aware':[cv.loc['nested_draw_aware',metric]],'Later reference':[test.loc['reference',metric]],'Later draw-aware':[test.loc['draw_aware',metric]]}).T
    ax.bar(range(4),data[0],color=['#728395','#278b90','#728395','#278b90']);ax.set_xticks(range(4),['CV\nreference','CV\ndraw-aware','Later\nreference','Later\ndraw-aware']);ax.set_ylim(0,.7);ax.set_title(metric.replace('_',' ').title())
    for i,v in enumerate(data[0]):ax.text(i,v+.015,f'{v:.1%}',ha='center',fontsize=10)
fig.suptitle('Two separate evaluation cohorts: compare models within each cohort',y=1.04);save('01_tradeoff.png')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,tag in zip(axs,['reference','draw_aware']):
    g=pred[pred.model==tag];cm=confusion_matrix(g.outcome,g.predicted,labels=['A','D','H'])
    sns.heatmap(cm,annot=True,fmt='d',cbar=False,cmap='Blues',xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'],ax=ax)
    ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(tag.replace('_',' ').title())
fig.suptitle('Reserved partial 2024–25 season: 214 fixtures',y=1.04);save('02_confusion.png')
fig,ax=plt.subplots(figsize=(9,5))
scatter=ax.scatter(search.accuracy,search.draw_recall,c=search.macro_f1,cmap='viridis',s=55)
ax.set(xlabel='Inner OOF accuracy',ylabel='Inner OOF draw recall',title='Final development search: each point is a model + decision rule')
ax.axhline(.1,color='grey',linestyle='--',label='10% draw-recall goal');ax.legend();fig.colorbar(scatter,label='Macro F1');save('03_search.png')
fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,c,label in zip(axs,['A','D','H'],['Away win','Draw','Home win']):
    ax.plot([0,1],[0,1],'--',color='gray')
    for tag in ['reference','draw_aware']:
        g=pred[pred.model==tag];a=g['p_'+c].to_numpy();y=(g.outcome==c).to_numpy();b=np.minimum((a*5).astype(int),4)
        points=[(a[b==i].mean(),y[b==i].mean(),sum(b==i)) for i in range(5) if sum(b==i)]
        ax.plot([v[0] for v in points],[v[1] for v in points],marker='o',label=tag)
        if tag=='draw_aware':
            for x,y,n in points:ax.annotate(str(n),(x,y),fontsize=8)
    ax.set(title=label,xlabel='Mean predicted probability',ylabel='Observed fraction',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
fig.suptitle('Later holdout calibration; numbers are forest bin counts',y=1.04);save('04_calibration.png')

def table(df):
    cols=['accuracy','macro_f1','draw_precision','draw_recall','log_loss']
    d=df[cols].reset_index();lines=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    for row in d.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)
sections=[('Can we improve accuracy and recognize draws?', '''We tested the idea rather than promising a higher score. The new random forest recognizes draws and improves macro F1. Its later holdout accuracy is 51.4% versus 50.0% for a refitted reference on exactly the same 214 matches—a gain of only three correct predictions. The approximate paired week-block 95% accuracy-difference interval is -3.1 to +6.1 percentage points, so this is not convincing evidence of an accuracy improvement.

In nested chronological validation the draw-aware selection procedure lost accuracy (47.2% versus 55.4%), while draw recall rose from 0% to 25.0%. No tested configuration satisfied both our declared development goals. Therefore the new forest is a documented draw-sensitive alternative, not a proven replacement for the original probability model. We did not keep adjusting settings after the new holdout was opened.''',['01_tradeoff.png']),
('Why the original model never chose a draw','''Draw probabilities were generally below the home-win or away-win probability. For example, probabilities [away=.36, draw=.30, home=.34] produce an away prediction even though a draw is plausible. A draw decision multiplier of 1.5 would choose draw using scores [.36,.45,.34], while the reported probabilities must remain [.36,.30,.34]. These scores are not calibrated probabilities.

Recognizing more draws can also produce more false draw predictions. The correct goal is not merely making the model say “draw”; it is showing the accuracy, precision, recall and probability-quality trade-offs on later data. Lecture 7 covers threshold-sensitive classification metrics and Lecture 9 covers logistic probabilities. The custom multiclass rule extends that idea; it is not a binary 0.5 cutoff.''',[]),
('What changed, and how leakage was prevented','''The first experiment is preserved. Its already-inspected 2023–24 test is now explicitly part of development, together with 2021–22 and 2022–23. A different, previously unevaluated set of 214 eligible completed 2024–25 fixtures is reserved for the final check, from August 16, 2024 through January 20, 2025. The snapshot has 219 completed fixtures that season; five lack enough prior team history. It is incomplete and not necessarily a contiguous league matchweek block; conclusions apply to these archived completed fixtures.

Ten outer chronological folds evaluate the complete selection procedure. Within each outer training set, three inner chronological folds choose both the model and the draw multiplier. All scaling and VIF pruning happen within the relevant training data. Date boundaries are saved and tested. No outer-validation outcome selects its own model or rule. Earlier observed match results may update later pre-match history, but never fitted coefficients within a fold.

The search includes eight regularized multinomial logistic configurations, two shallow trees and two forests. Logistic models compare signed differences with three additional pre-match inputs: absolute Elo gap, total recent goals scored, and total recent goals conceded. The aim is to represent team similarity and scoring environment; the result does not prove those features are beneficial. Logistic training draw weight is 1 or 1.5; tree/forest use balanced training weights. Each candidate also tries four decision multipliers, giving 48 model/rule combinations. No SMOTE or new advanced model family was added. Forest coverage in later lectures remains unverified locally, but it is part of the team's requested comparison.

The declared selector maximizes macro F1 among configurations with at least 10% inner draw recall and accuracy within two percentage points of the fixed reference. If none qualify, it reports failure and selects unconstrained macro F1 for a clearly labeled challenger. All ten outer searches and the final development search required that fallback. The originally chosen reference settings are fixed, although they were developed using some of these historical data; nested results are a diagnostic comparison, not fresh external evidence. The new later holdout is the separate final check.''',['03_search.png']),
('Nested chronological validation',table(cv)+'''

These scores pool 653 outer-validation fixtures. “Nested argmax” uses the same selected model probabilities without the draw multiplier; “nested draw aware” uses the independently selected rule in each fold. Probability metrics are identical between those two decision policies. The selection procedure improves macro F1 but fails to maintain accuracy. This is why simply reporting the best development-search score would have been misleading.''',[]),
('Reserved later-season results',table(test)+'''

The final development choice is a depth-4 random forest with minimum leaf size 20, 150 trees and balanced class weights. Its selected draw multiplier is 1.0: the final forest recognizes draws using ordinary argmax. Extra draw boosting was not selected. Thus the final draw improvement comes from the selected fitted model, not a post-test threshold change.

It predicts draw 27 times: 10 correct and 17 incorrect. There are 57 actual draws, giving draw recall 17.5% and draw precision 37.0%. It still misses 47 draws. Macro F1 improves from 0.3815 to 0.4668, while log loss worsens from 1.0027 to 1.0119. The approximate paired week-block interval for the macro-F1 difference is +0.0316 to +0.1402, conditional on these models and this partial season; team and cross-week dependence remain.

Do not compare 51.4% here directly with the previous 56.5%: they cover different seasons and training periods. The fair comparison here is 51.4% versus 50.0%. There is no full-season or live-2026 accuracy guarantee.''',['02_confusion.png','04_calibration.png']),
('What the team should present','''Present this as a measured trade-off, not an accuracy breakthrough. Explain why macro F1 was added, show the same-row baseline comparisons, and disclose that the accuracy constraint failed in development. Keep the original model for the probability-focused conclusion; show the forest as the draw-sensitive alternative. The forest does not dominate the reference across objectives.

Future improvements need richer verified pre-match information or more independent seasons. Do not keep using these 214 fixtures as an untouched test after seeing this result. If the professor requires one final model, state the objective explicitly before choosing: probability quality favors the original approach; equal attention to all three classes motivates the draw-sensitive alternative, with its documented accuracy risk.

Sources: the local Lecture 7 Q&A (thresholds and class metrics), Lecture 9 Q&A (logistic regression), and https://scikit-learn.org/1.6/modules/classification_threshold.html. The latter discusses separating probability estimation from label decisions and avoiding same-data fitting and threshold tuning. Our implementation uses custom chronological multiclass selection, not the library's default binary threshold tuner.''',[])]
md=[];parts=[];cells=[nbf.v4.new_markdown_cell('# Draw recognition follow-up\n\nExecuted review of a frozen second experiment. The v1 notebook and results remain unchanged.'),nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nR=Path.cwd()/'reports/draw_refinement'\nassert (R/'test_manifest.json').exists()")]
for title,body,figures in sections:
    md+=['## '+title,body]+[f'![{f}](figures/{f})' for f in figures]
    bodyhtml=''.join('<p>'+html.escape(p)+'</p>' if not p.startswith('|') else (cv if title.startswith('Nested') else test)[['accuracy','macro_f1','draw_precision','draw_recall','log_loss']].to_html(float_format=lambda v:f'{v:.4f}') for p in body.split('\n\n'))
    pics=''.join('<img alt="'+f+'" src="data:image/png;base64,'+base64.b64encode((F/f).read_bytes()).decode()+'">' for f in figures)
    parts.append('<section><h2>'+html.escape(title)+'</h2>'+bodyhtml+pics+'</section>')
    cells.append(nbf.v4.new_markdown_cell('## '+title+'\n\n'+body))
    if figures:cells.append(nbf.v4.new_code_cell('\n'.join(f"display(Image(filename=str(R/'figures/{f}')))" for f in figures)))
cells.append(nbf.v4.new_code_cell("display(pd.read_csv(R/'final_vif.csv'))\ndisplay(pd.read_csv(R/'outer_splits.csv'))\ndisplay(pd.read_csv(R/'test_metrics.csv'))"))
(R/'RESULTS.md').write_text('\n\n'.join(md)+'\n')
(R/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Draw recognition follow-up</title><style>body{max-width:1080px;margin:auto;padding:30px;background:#f4f6fa;color:#152b40;font:18px/1.6 system-ui}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto;margin-top:20px}table{font-size:14px;border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}</style><h1>Can we recognize draws without losing accuracy?</h1>'+''.join(parts)+'</html>')
nb=nbf.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'display_name':'Python (ISC project)','language':'python','name':'isc-project'}
nbf.write(nb,ROOT/'Draw_Recognition_Followup.ipynb')
print('Created follow-up report, plots and notebook')
