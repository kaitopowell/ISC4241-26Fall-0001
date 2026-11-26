"""Accuracy-focused follow-up; preserves the original experiment unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from analysis.data import ROOT, History, REDUCED
from analysis.models import FeatureSet, CLASSES, probabilities, metrics, forward_splits, vif_table

OUT=ROOT/'reports/expanded';DATA=ROOT/'data/expanded';ART=ROOT/'artifacts/expanded.joblib'
# Last-five features remain an explicit comparator. Longer history reduces form noise.
LONG=['gf20_diff','ga20_diff','ppg20_diff','elo_diff','rest_diff']
VENUE=LONG+['venue_gf_diff','venue_ga_diff']
SHOTS=VENUE+['sot20_diff','sot_against20_diff','shots20_diff','shots_against20_diff']

def load_matches():
    frames=[];manifest=[]
    for path in sorted(DATA.glob('E0_*.csv')):
        code=path.stem.split('_')[1];year=2000+int(code[:2])
        d=pd.read_csv(path)
        required=['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HS','AS','HST','AST']
        assert set(required)<=set(d),path.name
        d=d[required].dropna(how='all').copy()
        d.columns=['date','home','away','hg','ag','outcome','hs','as','hst','ast']
        d['date']=pd.to_datetime(d.date,dayfirst=True,format='mixed')
        d['season']=f'{year}-{year+1}';d['home_id']=d.home;d['away_id']=d.away
        assert len(d)==380 and d.home.nunique()==20 and d.away.nunique()==20
        assert d.home.value_counts().eq(19).all() and d.away.value_counts().eq(19).all()
        assert not d.duplicated(['home','away']).any()
        assert d[['date','home','away','hg','ag','outcome']].notna().all().all()
        assert (d[['hg','ag']]>=0).all().all() and (d[['hg','ag']]%1==0).all().all()
        expected=np.where(d.hg>d.ag,'H',np.where(d.hg<d.ag,'A','D'))
        assert np.array_equal(expected,d.outcome)
        d['match_id']=d.season+'_'+d.home+'_'+d.away
        frames.append(d)
        manifest.append({'file':path.name,'url':f'https://www.football-data.co.uk/mmz4281/{code}/E0.csv','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(d)})
    return pd.concat(frames).sort_values(['date','match_id']).reset_index(drop=True),manifest

def features(matches):
    history=History();extra={};rows=[];excluded=[]
    def avg(records,j):
        a=np.array([r[j] for r in records],float)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
    for date,day in matches.groupby('date',sort=True):
        for r in day.itertuples():
            f=history.features(r.home,r.away,date)
            if f is None:excluded.append({'match_id':r.match_id,'season':r.season});continue
            stats={}
            for side,team,venue in [('h',r.home,'H'),('a',r.away,'A')]:
                past=history.games[team][-20:]
                stats[side]={'ppg20':np.mean([g[1] for g in past]),'gf20':np.mean([g[2] for g in past]),'ga20':np.mean([g[3] for g in past])}
                records=extra[team][-20:]
                for stat,j in [('shots20',1),('shots_against20',2),('sot20',3),('sot_against20',4)]:stats[side][stat]=avg(records,j)
                vs=[g for g in extra[team] if g[0]==venue][-10:]
                stats[side]['venue_gf']=avg(vs,5);stats[side]['venue_ga']=avg(vs,6)
            f.update({k+'_diff':stats['h'][k]-stats['a'][k] for k in stats['h']})
            rows.append({**r._asdict(),**f})
        # Raw current-match statistics are only added AFTER every same-day prediction.
        for r in day.to_dict('records'):
            extra.setdefault(r['home'],[]).append(('H',r['hs'],r['as'],r['hst'],r['ast'],r['hg'],r['ag']))
            extra.setdefault(r['away'],[]).append(('A',r['as'],r['hs'],r['ast'],r['hst'],r['ag'],r['hg']))
        history.update_day(day)
    return pd.DataFrame(rows).drop(columns='Index',errors='ignore'),pd.DataFrame(excluded)

def pipeline(model,cols,prune=False):
    # Feature selection's VIF is used only for complete score features; optional
    # shot statistics are median-imputed fold-locally, then regularized, not VIF-pruned.
    return Pipeline([('features',FeatureSet(columns=cols,prune=prune)),('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',model)])

def grid():
    yield 'prior','baseline',pipeline(DummyClassifier(strategy='prior'),LONG)
    yield 'original_recipe','baseline',pipeline(LogisticRegression(C=.01,max_iter=3000),REDUCED,True)
    for label,cols in [('elo',['elo_diff']),('long',LONG),('venue',VENUE),('shots',SHOTS)]:
        for c in [.01,.1,1,10]:yield f'logistic_{label}_C{c}','logistic',pipeline(LogisticRegression(C=c,max_iter=3000),cols)
    # Added after development-only VIF diagnosis, before opening the new season.
    for c in [.01,.1,1,10]:
        pruned=Pipeline([('features',FeatureSet(columns=SHOTS,prune=False).set_output(transform='pandas')),
                         ('impute',SimpleImputer(strategy='median').set_output(transform='pandas')),
                         ('vif',FeatureSet(columns=SHOTS,prune=True)),
                         ('scale',StandardScaler()),('model',LogisticRegression(C=c,max_iter=3000))])
        yield f'logistic_shots_pruned_C{c}','logistic',pruned
    for cols,label in [(VENUE,'venue'),(SHOTS,'shots')]:
        for depth in [3,6]:
            for leaf in [20,60]:yield f'tree_{label}_d{depth}_n{leaf}','tree',pipeline(DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=42),cols)
        for depth in [4,8,None]:
            for leaf in [10,30]:yield f'forest_{label}_d{depth}_n{leaf}','forest',pipeline(RandomForestClassifier(n_estimators=200,max_depth=depth,min_samples_leaf=leaf,max_features=.8,n_jobs=1,random_state=42),cols)

def fingerprint():
    paths=[Path(__file__),ROOT/'analysis/data.py',ROOT/'analysis/models.py']+sorted(DATA.glob('E0_*.csv'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()

def develop():
    matches,manifest=load_matches();f,excluded=features(matches)
    OUT.mkdir(parents=True,exist_ok=True)
    (DATA/'provenance.json').write_text(json.dumps({'source':'Football-Data.co.uk','retrieved':'2026-11-26','files':manifest,'usage':'Public historical CSVs; source attribution does not grant a new license.'},indent=2))
    excluded.to_csv(OUT/'warmup_exclusions.csv',index=False)
    f.to_csv(DATA/'features.csv',index=False)
    # 2014-16 warm-up builds stronger priors; 2016-24 training includes the already-opened old test.
    dev=f[(f.season>='2016-2017')&(f.season<'2024-2025')].reset_index(drop=True)
    splits=list(forward_splits(dev,10))
    summaries=[];foldrows=[];oofs={};models={};splitrows=[]
    for fold,tr,va in splits:splitrows.append({'fold':fold,'train_n':len(tr),'validation_n':len(va),'train_end':str(dev.iloc[tr].date.max().date()),'validation_start':str(dev.iloc[va].date.min().date()),'validation_end':str(dev.iloc[va].date.max().date())})
    pd.DataFrame(splitrows).to_csv(OUT/'splits.csv',index=False)
    for name,family,pipe in grid():
        print('Validating',name,flush=True);blocks=[]
        for fold,tr,va in splits:
            m=clone(pipe).fit(dev.iloc[tr],dev.iloc[tr].outcome);p=probabilities(m,dev.iloc[va])
            foldrows.append({'name':name,'fold':fold,**metrics(dev.iloc[va].outcome,p),'train_accuracy':metrics(dev.iloc[tr].outcome,probabilities(m,dev.iloc[tr]))['accuracy']})
            b=dev.iloc[va][['match_id','date','home','away','outcome']].copy();b['fold']=fold
            for j,c in enumerate(CLASSES):b['p_'+c]=p[:,j]
            blocks.append(b)
        oof=pd.concat(blocks,ignore_index=True);oofs[name]=oof;models[name]=pipe
        summaries.append({'name':name,'family':family,**metrics(oof.outcome,oof[['p_A','p_D','p_H']].to_numpy())})
    summary=pd.DataFrame(summaries).sort_values(['accuracy','log_loss'],ascending=[False,True])
    summary.to_csv(OUT/'cv_summary.csv',index=False);pd.DataFrame(foldrows).to_csv(OUT/'fold_metrics.csv',index=False)
    selected=summary.iloc[0]['name']
    # Freeze ONE accuracy-selected model plus prior/old-recipe comparators before new evaluation.
    names=list(dict.fromkeys([selected,'original_recipe','prior']))
    fitted={name:clone(models[name]).fit(dev,dev.outcome) for name in names}
    for name in names:oofs[name].to_csv(OUT/f'oof_{name}.csv',index=False)
    lock={'selected':selected,'criterion':'highest pooled chronological OOF accuracy; tie: lower log loss','development_seasons':'2016-2017 through 2023-2024','warmup_seasons':'2014-2015 and 2015-2016','test_season':'2024-2025','development_n':len(dev),'candidates':len(summary),'fingerprint':fingerprint(),'features':(fitted[selected].named_steps['vif'].kept_ if 'vif' in fitted[selected].named_steps else fitted[selected].named_steps['features'].kept_),'note':'Follow-up initiated after seeing original 2023-24 test. It is now development data. 2024-25 is new evaluation, never used for selection. Accuracy is the revised objective requested by the user.'}
    (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2));joblib.dump(fitted,ART)
    print(summary.head(10).to_string(index=False),flush=True)

def evaluate():
    lock=json.loads((OUT/'selection_lock.json').read_text());assert lock['fingerprint']==fingerprint()
    manifest=OUT/'test_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==lock,'Cannot overwrite evaluated configuration'
    x=pd.read_csv(DATA/'features.csv',parse_dates=['date']);x=x[x.season==lock['test_season']]
    fitted=joblib.load(ART);rows=[]
    for name,m in fitted.items():
        p=probabilities(m,x);rows.append({'name':name,**metrics(x.outcome,p)})
        b=x[['match_id','date','home','away','outcome']].copy()
        for j,c in enumerate(CLASSES):b['p_'+c]=p[:,j]
        b['predicted']=CLASSES[p.argmax(axis=1)];b.to_csv(OUT/f'test_{name}.csv',index=False)
    result=pd.DataFrame(rows)
    if (OUT/'test_metrics.csv').exists():pd.testing.assert_frame_equal(pd.read_csv(OUT/'test_metrics.csv'),result,check_exact=False,rtol=1e-8)
    result.to_csv(OUT/'test_metrics.csv',index=False);manifest.write_text(json.dumps(lock,indent=2))
    print(result.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['develop','test'],required=True)
    args=p.parse_args();develop() if args.stage=='develop' else evaluate()
