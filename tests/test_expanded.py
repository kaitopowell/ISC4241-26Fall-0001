import numpy as np
import pandas as pd
from experiments.improve import load_matches,features,SHOTS,grid
from analysis.models import probabilities

def test_expanded_seasons_and_target():
    matches,manifest=load_matches()
    assert len(matches)==4180 and len(manifest)==11
    assert (matches.groupby('season').size()==380).all()

def test_future_shots_and_scores_do_not_leak():
    matches,_=load_matches();matches=matches[matches.season<'2017-2018']
    f,_=features(matches)
    date=pd.Timestamp('2016-03-01')
    changed=matches.copy();changed.loc[changed.date>=date,['hg','ag','hs','as','hst','ast']]=[7,0,90,1,80,1]
    g,_=features(changed)
    pd.testing.assert_frame_equal(f.loc[f.date<=date,SHOTS],g.loc[g.date<=date,SHOTS])

def test_optional_stats_imputed_only_from_training():
    matches,_=load_matches();f,_=features(matches[matches.season<'2016-2017'])
    pipe=next(p for name,_,p in grid() if name=='logistic_shots_C0.1')
    train=f.iloc[:200].copy();train.loc[train.index[:10],'sot20_diff']=np.nan
    pipe.fit(train,train.outcome)
    learned=pipe.named_steps['impute'].statistics_.copy()
    valid=f.iloc[200:220].copy();valid['sot20_diff']=np.nan
    p=probabilities(pipe,valid)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
    np.testing.assert_array_equal(learned,pipe.named_steps['impute'].statistics_)

def test_pruning_after_imputation_preserves_names_and_bounds_vif():
    from analysis.models import vif_table
    matches,_=load_matches();f,_=features(matches[matches.season<'2016-2017'])
    p=next(p for name,_,p in grid() if name=='logistic_shots_pruned_C0.1')
    x=f.iloc[:300].copy();x.loc[x.index[:5],'sot20_diff']=np.nan
    p.fit(x,x.outcome)
    imputed=p.named_steps['impute'].transform(p.named_steps['features'].transform(x))
    assert vif_table(imputed[p.named_steps['vif'].kept_]).vif.max()<=5
    assert np.isfinite(probabilities(p,x)).all()
