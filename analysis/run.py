"""Run development first, then freeze selection and evaluate the held-out season."""
import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from .data import ROOT, FULL, REDUCED, extract_cached, clean_fixtures, build_features
from .models import candidates, forward_splits, probabilities, metrics, vif_table, CLASSES
from . import plots

REPORT = ROOT/'reports'
ART = ROOT/'artifacts'

def fingerprint():
    paths = sorted((ROOT/'analysis').glob('*.py'))+[ROOT/'data/raw/cached_fixtures.csv']
    h=hashlib.sha256()
    for p in paths:h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def prepare():
    raw=extract_cached()
    fixtures,audit,coverage=clean_fixtures(raw)
    features,excluded,history=build_features(fixtures)
    test_season=sorted(fixtures.season.unique())[-1]
    dev=features[features.season<test_season].reset_index(drop=True)
    test=features[features.season==test_season].reset_index(drop=True)
    REPORT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True)
    audit.to_csv(REPORT/'cleaning_audit.csv',index=False)
    coverage.to_csv(REPORT/'season_coverage.csv',index=False)
    excluded.to_csv(REPORT/'insufficient_history.csv',index=False)
    features.to_csv(ROOT/'data/model_features.csv',index=False)
    fixtures.to_csv(ROOT/'data/completed_matches.csv',index=False)
    # Raw numerical outliers are retained, with an explicit list for review.
    d=fixtures[fixtures.season<test_season]
    d[(d.hg+d.ag>=7)|(abs(d.hg-d.ag)>=5)].to_csv(REPORT/'unusual_scorelines_development.csv',index=False)
    return fixtures,dev,test,history

def develop():
    fixtures,dev,test,history=prepare()
    splits=list(forward_splits(dev,10))
    split_rows=[]
    for fold,tr,va in splits:
        split_rows.append({'fold':fold,'train_n':len(tr),'validation_n':len(va),
                           'train_end':str(dev.iloc[tr].date.max().date()),
                           'validation_start':str(dev.iloc[va].date.min().date()),
                           'validation_end':str(dev.iloc[va].date.max().date())})
    pd.DataFrame(split_rows).to_csv(REPORT/'cv_splits.csv',index=False)
    # Collinearity diagnosis is visible on the INITIAL training window before refinement.
    initial=dev.iloc[splits[0][1]]
    vif_table(initial[FULL]).to_csv(REPORT/'initial_window_vif_full.csv',index=False)
    vif_table(initial[REDUCED]).to_csv(REPORT/'initial_window_vif_differences.csv',index=False)
    plots.exploration(fixtures[fixtures.season<test.season.iloc[0]],dev,splits)
    all_folds=[]; summaries=[]; predictions={}; pipelines={}; removals=[]
    for name,family,pipeline in candidates():
        print(f'Validating {name} on 10 chronological folds',flush=True)
        blocks=[];fold_losses=[]
        for fold,tr,va in splits:
            model=clone(pipeline)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(dev.iloc[tr],dev.iloc[tr].outcome)
            p=probabilities(model,dev.iloc[va]); m=metrics(dev.iloc[va].outcome,p)
            train=metrics(dev.iloc[tr].outcome,probabilities(model,dev.iloc[tr]))
            all_folds.append({'name':name,'family':family,'fold':fold,**m,'train_log_loss':train['log_loss'],'train_accuracy':train['accuracy']})
            for r in model.named_steps['features'].removed_:removals.append({'name':name,'fold':fold,**r})
            block=dev.iloc[va][['match_id','date','home','away','outcome']].copy()
            block['fold']=fold
            for j,c in enumerate(CLASSES):block[f'p_{c}']=p[:,j]
            blocks.append(block);fold_losses.append(m['log_loss'])
        oof=pd.concat(blocks,ignore_index=True)
        m=metrics(oof.outcome,oof[['p_A','p_D','p_H']].to_numpy())
        summaries.append({'name':name,'family':family,**{'cv_'+k:v for k,v in m.items()},'fold_log_loss_sd':np.std(fold_losses,ddof=1)})
        predictions[name]=oof; pipelines[name]=pipeline
    summary=pd.DataFrame(summaries).sort_values(['cv_log_loss','cv_macro_f1'],ascending=[True,False])
    folds=pd.DataFrame(all_folds)
    summary.to_csv(REPORT/'cv_summary.csv',index=False);folds.to_csv(REPORT/'cv_fold_metrics.csv',index=False)
    pd.DataFrame(removals).to_csv(REPORT/'fold_local_vif_removals.csv',index=False)
    champions=summary.loc[summary.groupby('family').cv_log_loss.idxmin()].sort_values('cv_log_loss')
    selected=str(summary.iloc[0]['name'])
    # Compare the strongest eligible candidate, even if the simple prior wins.
    fitted={}
    for name in champions.name:
        fitted[name]=clone(pipelines[name]).fit(dev,dev.outcome)
        predictions[name].to_csv(REPORT/f'oof_{name}.csv',index=False)
    refined=str(champions[champions.family=='logistic_refined'].iloc[0]['name'])
    tree_name=str(champions[champions.family=='decision_tree'].iloc[0]['name'])
    pruner=fitted[refined].named_steps['features']
    vif_table(dev[pruner.kept_]).to_csv(REPORT/'final_development_vif.csv',index=False)
    plots.heatmap(dev[pruner.kept_],'09_final_correlation','Refined logistic predictors: all development data')
    plots.comparison(summary,folds)
    for key,name in [('10_initial_oof','logistic_initial'),('11_refined_oof',refined),('12_selected_oof',selected)]:
        oof=predictions[name];plots.probability_diagnostics(oof,oof[['p_A','p_D','p_H']].to_numpy(),key,f'Development OOF: {name}')
    plots.interpretation(fitted[refined],fitted[tree_name],dev)
    lock={'selected_model':selected,'refined_logistic':refined,'champions':champions.name.tolist(),
          'primary_metric':'pooled chronological OOF multiclass log loss (lower is better)',
          'selection_note':'CV reused for tuning/selection, not an unbiased nested-CV performance estimate. Test season provides final assessment.',
          'test_season':str(test.season.iloc[0]),'development_rows':len(dev),'test_rows':len(test),
          'folds':split_rows,'refined_features':pruner.kept_,'fingerprint':fingerprint(),
          'python':platform.python_version(),'sklearn':sklearn.__version__,
          'elo':'Supplemental feature: start 1500, K=20, home offset=60; fixed before validation, not tuned. Includes no-Elo ablation.',
          'seed':42}
    (REPORT/'selection_lock.json').write_text(json.dumps(lock,indent=2))
    joblib.dump({'models':fitted,'lock':lock,'history':history,'team_names':dict(zip(fixtures.home_id,fixtures.home))},ART/'development_bundle.joblib')
    print(summary[['name','cv_log_loss','cv_macro_f1']].to_string(index=False),flush=True)
    print('Selection frozen. Run tests, inspect development diagnostics, then run --stage test.',flush=True)

def evaluate():
    lock=json.loads((REPORT/'selection_lock.json').read_text())
    if fingerprint()!=lock['fingerprint']:raise RuntimeError('Code/data changed after selection. Rerun development before testing.')
    manifest=REPORT/'test_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=lock['fingerprint']:
        raise RuntimeError('Test was already opened with different code/data. Preserve the original experiment; do not retune on this test.')
    if (REPORT/'test_metrics.csv').exists() and not manifest.exists():
        raise RuntimeError('Existing test results have no matching manifest; refusing to overwrite.')
    bundle=joblib.load(ART/'development_bundle.joblib')
    features=pd.read_csv(ROOT/'data/model_features.csv',parse_dates=['date'])
    test=features[features.season==lock['test_season']].reset_index(drop=True)
    rows=[]
    for name,model in bundle['models'].items():
        p=probabilities(model,test)
        rows.append({'name':name,'selected_by_cv':name==lock['selected_model'],**metrics(test.outcome,p)})
        out=test[['match_id','date','home','away','outcome']].copy()
        for j,c in enumerate(CLASSES):out[f'p_{c}']=p[:,j]
        out['predicted']=CLASSES[p.argmax(axis=1)]
        out.to_csv(REPORT/f'test_predictions_{name}.csv',index=False)
        if name==lock['selected_model']:
            plots.probability_diagnostics(test,p,'20_test',f'Untuned test {lock["test_season"]}: {name}')
    results=pd.DataFrame(rows)
    if (REPORT/'test_metrics.csv').exists():
        previous=pd.read_csv(REPORT/'test_metrics.csv')
        pd.testing.assert_frame_equal(previous,results,check_exact=False,rtol=1e-8,atol=1e-10)
        print('Reproduction matches the original held-out metrics.')
    results.to_csv(REPORT/'test_metrics.csv',index=False)
    manifest.write_text(json.dumps({'fingerprint':lock['fingerprint'],'selected_model':lock['selected_model']},indent=2))
    # Paired week-block bootstrap of loss differences: held-out comparison, not model selection.
    sel=pd.read_csv(REPORT/f'test_predictions_{lock["selected_model"]}.csv',parse_dates=['date'])
    ref=pd.read_csv(REPORT/'test_predictions_prior.csv')
    inds=np.array([list(CLASSES).index(c) for c in sel.outcome])
    psel=sel[['p_A','p_D','p_H']].to_numpy();pref=ref[['p_A','p_D','p_H']].to_numpy()
    diff=-np.log(np.clip(psel[np.arange(len(sel)),inds],1e-15,1))+np.log(np.clip(pref[np.arange(len(sel)),inds],1e-15,1))
    groups=sel.date.dt.to_period('W').astype(str).to_numpy();unique=np.unique(groups)
    blocks=[diff[groups==g] for g in unique];rng=np.random.default_rng(42)
    boots=[np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))]).mean() for _ in range(2000)]
    uncertainty={'comparison':'selected minus prior mean log loss; negative favors selected',
                 'estimate':float(diff.mean()),'week_block_percentile_95':np.quantile(boots,[.025,.975]).tolist(),
                 'limitation':'Approximate interval; week blocks may retain cross-week/team dependence. Conditional on these fitted models.'}
    (REPORT/'test_uncertainty.json').write_text(json.dumps(uncertainty,indent=2))
    print(results.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['develop','test'],required=True)
    args=parser.parse_args();develop() if args.stage=='develop' else evaluate()
