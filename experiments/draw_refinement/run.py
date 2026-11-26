"""Nested chronological selection; keeps the original experiment immutable."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
from analysis.data import ROOT,build_features
from analysis.models import make_pipeline,forward_splits,probabilities,metrics,CLASSES,vif_table

OUT=ROOT/'reports/draw_refinement'
BASE=['gf_diff','ga_diff','win_diff','rest_diff','elo_diff']
EXTRA=BASE+['abs_elo_gap','total_gf','total_ga']
MULTIPLIERS=[1.,1.25,1.5,1.75]

def fingerprint():
    paths=sorted((ROOT/'analysis').glob('*.py'))+sorted((ROOT/'experiments/draw_refinement').glob('*.py'))+[ROOT/'experiments/draw_refinement/PLAN.md',ROOT/'data/raw/cached_fixtures.csv']
    h=hashlib.sha256()
    for p in paths:h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def dataset():
    raw=pd.read_csv(ROOT/'data/raw/cached_fixtures.csv',dtype={'home_id':str,'away_id':str,'match_id':str,'league_id':str})
    raw=raw[(raw.status=='Match Finished')&(raw.league_id=='4328')].copy()
    if raw.match_id.isna().any() or raw.match_id.duplicated().any():raise ValueError('Invalid/duplicate match identifiers')
    raw['date']=pd.to_datetime(raw.date,errors='raise').dt.normalize()
    required=['date','season','home_id','away_id','home','away','hg','ag']
    if raw[required].isna().any().any():raise ValueError('Missing completed match fields')
    if (raw.home_id==raw.away_id).any() or raw.duplicated(['season','home_id','away_id']).any():raise ValueError('Invalid fixture')
    for c in ['hg','ag']:
        if not ((raw[c]>=0)&(raw[c]%1==0)).all():raise ValueError('Invalid score')
        raw[c]=raw[c].astype(int)
    if raw.groupby(['date','home_id']).size().max()>1:raise ValueError('Multiple home fixtures in a day')
    raw=raw.sort_values(['date','match_id']).reset_index(drop=True)
    raw['outcome']=np.where(raw.hg>raw.ag,'H',np.where(raw.hg<raw.ag,'A','D'))
    x,excluded,_=build_features(raw)
    x['abs_elo_gap']=x.elo_diff.abs()
    x['total_gf']=x.home_gf5+x.away_gf5
    x['total_ga']=x.home_ga5+x.away_ga5
    dev=x[x.season<'2024-2025'].reset_index(drop=True)
    test=x[x.season=='2024-2025'].reset_index(drop=True)
    assert dev.date.max()<test.date.min()
    return dev,test,excluded

def candidates():
    out={}
    for kind,cols in [('signed',BASE),('symmetric',EXTRA)]:
        for c in [.1,1.]:
            for w in [1.,1.5]:
                out[f'logistic_{kind}_C{c}_draw{w}']=make_pipeline(LogisticRegression(C=c,class_weight={'A':1.,'D':w,'H':1.},max_iter=3000),cols,True)
    for depth,leaf in [(2,30),(4,30)]:
        out[f'tree_d{depth}']=make_pipeline(DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=42),EXTRA,True)
    for depth,leaf in [(4,20),(8,25)]:
        out[f'forest_d{depth}']=make_pipeline(RandomForestClassifier(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=42,n_jobs=1),EXTRA,True)
    return out

def reference():return make_pipeline(LogisticRegression(C=.01,max_iter=3000))

def labels(p,multiplier=1.):
    scores=np.asarray(p).copy();scores[:,1]*=multiplier
    return CLASSES[scores.argmax(axis=1)]

def score(y,p,multiplier=1.):
    result=metrics(y,p);pred=labels(p,multiplier)
    pr,re,f1,support=precision_recall_fscore_support(y,pred,labels=CLASSES,zero_division=0)
    result.update(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=CLASSES,average='macro',zero_division=0),balanced_accuracy=float(re.mean()),draw_precision=pr[1],draw_recall=re[1],draw_f1=f1[1],predicted_draws=int(np.sum(pred=='D')))
    return result

def choose(table,reference_accuracy):
    table=table.copy()
    table['eligible']=(table.accuracy>=reference_accuracy-.02)&(table.draw_recall>=.1)
    pool=table[table.eligible]
    fallback=pool.empty
    if fallback:pool=table
    winner=pool.sort_values(['macro_f1','accuracy','log_loss','name','multiplier'],ascending=[False,False,True,True,True]).iloc[0]
    return winner,table,fallback

def tune(train):
    splits=list(forward_splits(train,3));models=candidates();models['reference']=reference()
    rows=[];refs=[]
    for name,pipe in models.items():
        ys=[];ps=[]
        for _,tr,va in splits:
            model=clone(pipe).fit(train.iloc[tr],train.iloc[tr].outcome)
            ys.extend(train.iloc[va].outcome);ps.append(probabilities(model,train.iloc[va]))
        p=np.vstack(ps);y=np.array(ys)
        if name=='reference':ref_accuracy=score(y,p)['accuracy']
        else:
            for m in MULTIPLIERS:rows.append({'name':name,'multiplier':m,**score(y,p,m)})
    return choose(pd.DataFrame(rows),ref_accuracy)

def block(frame,p,tag,multiplier):
    result=frame[['match_id','date','home','away','outcome']].copy()
    for j,c in enumerate(CLASSES):result['p_'+c]=p[:,j]
    result['predicted']=labels(p,multiplier);result['multiplier']=multiplier;result['model']=tag
    return result

def develop():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'test_manifest.json').exists():raise RuntimeError('Follow-up already tested. Preserve it; no further development on this holdout.')
    dev,test,excluded=dataset();excluded.to_csv(OUT/'warmup_exclusions.csv',index=False)
    allrows=[];foldrows=[];boundaries=[];innerbounds=[]
    for fold,tr,va in forward_splits(dev,10):
        train=dev.iloc[tr].reset_index(drop=True);valid=dev.iloc[va]
        for inner,itr,iva in forward_splits(train,3):
            innerbounds.append({'outer_fold':fold,'inner_fold':inner,'train_end':str(train.iloc[itr].date.max().date()),'valid_start':str(train.iloc[iva].date.min().date()),'valid_end':str(train.iloc[iva].date.max().date())})
        winner,table,fallback=tune(train)
        table.to_csv(OUT/f'inner_search_outer_{fold}.csv',index=False)
        model=clone(candidates()[winner['name']]).fit(train,train.outcome)
        p=probabilities(model,valid)
        ref=reference().fit(train,train.outcome);rp=probabilities(ref,valid)
        for tag,probs,m in [('nested_draw_aware',p,float(winner.multiplier)),('nested_argmax',p,1.),('reference',rp,1.)]:
            b=block(valid,probs,winner['name'] if tag!='reference' else tag,m);b['approach']=tag;b['fold']=fold;allrows.append(b)
            foldrows.append({'fold':fold,'approach':tag,'selected_model':winner['name'],'constraints_unmet':fallback,**score(valid.outcome,probs,m)})
        boundaries.append({'fold':fold,'train_n':len(train),'valid_n':len(valid),'train_end':str(train.date.max().date()),'valid_start':str(valid.date.min().date()),'valid_end':str(valid.date.max().date()),'selected_model':winner['name'],'multiplier':float(winner.multiplier),'constraints_unmet':fallback})
        print(f'Outer {fold}/10: {winner["name"]}, draw multiplier {winner.multiplier}',flush=True)
    predictions=pd.concat(allrows,ignore_index=True);predictions.to_csv(OUT/'nested_predictions.csv',index=False)
    pd.DataFrame(foldrows).to_csv(OUT/'nested_fold_metrics.csv',index=False)
    pd.DataFrame(boundaries).to_csv(OUT/'outer_splits.csv',index=False)
    pd.DataFrame(innerbounds).to_csv(OUT/'inner_splits.csv',index=False)
    summary=[]
    for tag,g in predictions.groupby('approach'):
        # Decisions have different per-fold multipliers; recompute label metrics directly.
        m=score(g.outcome,g[['p_A','p_D','p_H']].to_numpy())
        pr,re,f1,_=precision_recall_fscore_support(g.outcome,g.predicted,labels=CLASSES,zero_division=0)
        m.update(accuracy=accuracy_score(g.outcome,g.predicted),macro_f1=float(f1.mean()),balanced_accuracy=float(re.mean()),draw_precision=pr[1],draw_recall=re[1],draw_f1=f1[1],predicted_draws=int((g.predicted=='D').sum()))
        summary.append({'approach':tag,**m})
    pd.DataFrame(summary).to_csv(OUT/'nested_summary.csv',index=False)
    winner,search,fallback=tune(dev);search.to_csv(OUT/'final_development_search.csv',index=False)
    model=clone(candidates()[winner['name']]).fit(dev,dev.outcome);ref=reference().fit(dev,dev.outcome)
    vif_table(dev[model.named_steps['features'].kept_]).to_csv(OUT/'final_vif.csv',index=False)
    lock={'fingerprint':fingerprint(),'selected_model':winner['name'],'multiplier':float(winner.multiplier),'constraints_unmet':fallback,'development_n':len(dev),'test_n':len(test),'test_start':str(test.date.min().date()),'test_end':str(test.date.max().date()),'features':model.named_steps['features'].kept_,'primary_objective':'macro F1, subject to inner accuracy within 2pp of reference and draw recall >=10%; fallback unconstrained macro F1'}
    (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2))
    joblib.dump({'model':model,'reference':ref,'lock':lock},ROOT/'artifacts/draw_refinement.joblib')
    print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(lock,indent=2))

def evaluate():
    lock=json.loads((OUT/'selection_lock.json').read_text())
    if lock['fingerprint']!=fingerprint():raise RuntimeError('Changed experiment after freeze')
    if (OUT/'test_manifest.json').exists():raise RuntimeError('Already evaluated; inspect saved result instead of tuning further.')
    bundle=joblib.load(ROOT/'artifacts/draw_refinement.joblib');_,test,_=dataset()
    p=probabilities(bundle['model'],test);rp=probabilities(bundle['reference'],test)
    rows=[];frames=[]
    for tag,probs,m in [('draw_aware',p,lock['multiplier']),('same_model_argmax',p,1.),('reference',rp,1.)]:
        frames.append(block(test,probs,tag,m));rows.append({'approach':tag,**score(test.outcome,probs,m)})
    pd.DataFrame(rows).to_csv(OUT/'test_metrics.csv',index=False)
    pd.concat(frames,ignore_index=True).to_csv(OUT/'test_predictions.csv',index=False)
    # Paired resampling of whole weeks for accuracy and macro-F1 differences.
    rng=np.random.default_rng(42);groups=test.date.dt.to_period('W').astype(str).to_numpy()
    blocks=[np.flatnonzero(groups==w) for w in np.unique(groups)]
    y=test.outcome.to_numpy();chosen=labels(p,lock['multiplier']);baseline=labels(rp)
    diffs=[]
    for _ in range(2000):
        ix=np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))])
        diffs.append([np.mean(chosen[ix]==y[ix])-np.mean(baseline[ix]==y[ix]),f1_score(y[ix],chosen[ix],labels=CLASSES,average='macro',zero_division=0)-f1_score(y[ix],baseline[ix],labels=CLASSES,average='macro',zero_division=0)])
    uncertainty={'comparison':'draw-aware minus reference; positive favors draw-aware','accuracy_95':np.quantile(np.array(diffs)[:,0],[.025,.975]).tolist(),'macro_f1_95':np.quantile(np.array(diffs)[:,1],[.025,.975]).tolist(),'limitation':'Paired week-block percentile intervals, conditional on fitted models; cross-week/team dependence and partial-season bias remain.'}
    (OUT/'uncertainty.json').write_text(json.dumps(uncertainty,indent=2))
    (OUT/'test_manifest.json').write_text(json.dumps(lock,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(uncertainty)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['develop','test'],required=True)
    args=parser.parse_args();develop() if args.stage=='develop' else evaluate()
