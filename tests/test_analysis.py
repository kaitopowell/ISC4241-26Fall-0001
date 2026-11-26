import numpy as np
import pandas as pd
import pytest
from analysis.data import extract_cached,clean_fixtures,build_features,FULL,REDUCED
from analysis.models import forward_splits,FeatureSet,metrics,CLASSES

@pytest.fixture(scope='module')
def clean():return clean_fixtures(extract_cached())[0]

def test_source_completeness(clean):
    assert clean.groupby('season').size().to_dict()=={'2021-2022':380,'2022-2023':380,'2023-2024':380}
    assert not clean.match_id.duplicated().any()

def test_current_and_future_scores_never_change_prematch_features(clean):
    original,_,_=build_features(clean)
    date=original.date.unique()[30]
    modified=clean.copy()
    modified.loc[modified.date>=date,['hg','ag']]=[8,0]
    changed,_,_=build_features(modified)
    cols=list(dict.fromkeys(FULL+REDUCED))
    a=original[original.date<=date].set_index('match_id')[cols]
    b=changed[changed.date<=date].set_index('match_id')[cols]
    pd.testing.assert_frame_equal(a,b)

def test_same_day_order_does_not_change_features(clean):
    first,_,_=build_features(clean)
    shuffled=clean.groupby('date',group_keys=False).sample(frac=1,random_state=3).sort_values('date',kind='stable')
    second,_,_=build_features(shuffled)
    cols=list(dict.fromkeys(FULL+REDUCED))
    pd.testing.assert_frame_equal(first.set_index('match_id')[cols].sort_index(),second.set_index('match_id')[cols].sort_index())

def test_ten_folds_are_strictly_forward(clean):
    f,_,_=build_features(clean);dev=f[f.season<'2023-2024'].reset_index(drop=True)
    folds=list(forward_splits(dev));assert len(folds)==10
    used=[]
    for _,tr,va in folds:
        assert dev.iloc[tr].date.max()<dev.iloc[va].date.min()
        assert set(tr).isdisjoint(va)
        assert (dev.iloc[va].season<'2023-2024').all()
        used.extend(va)
    assert len(set(used))==len(used)

def test_vif_selection_learns_only_fit_rows():
    rng=np.random.default_rng(42);a=rng.normal(size=(100,2));x=pd.DataFrame({'a':a[:,0],'b':a[:,1],'c':a[:,0]+a[:,1]})
    pruner=FeatureSet(columns=['a','b','c'],prune=True).fit(x)
    assert len(pruner.kept_)==2
    kept=pruner.kept_.copy();pruner.transform(x*10000);assert pruner.kept_==kept

def test_perfect_probabilities_and_class_order():
    m=metrics(CLASSES,np.eye(3));assert m['accuracy']==1 and m['brier_multiclass']==0
    assert m['log_loss']<1e-12

def test_conflicting_duplicate_rejected(clean):
    duplicate=clean.iloc[[0]].copy();duplicate['hg']+=1
    with pytest.raises(ValueError,match='Conflicting'):
        clean_fixtures(pd.concat([clean,duplicate],ignore_index=True))

def test_draw_history_and_hand_computed_features():
    from analysis.data import History
    h=History()
    for date in pd.date_range('2023-01-01',periods=5):
        h.update_day(pd.DataFrame([{'home_id':'a','away_id':'b','hg':2,'ag':2,'date':date}]))
    f=h.features('a','b',pd.Timestamp('2023-01-06'))
    assert f['home_ppg5']==1 and f['away_ppg5']==1
    assert f['home_win5']==0 and f['home_gf5']==2 and f['home_ga5']==2
    assert f['home_rest']==1 and f['gf_diff']==0

def test_pipeline_cannot_use_current_outcome_columns(clean):
    from analysis.models import candidates,probabilities
    f,_,_=build_features(clean)
    train=f.iloc[:266];valid=f.iloc[266:280]
    for name,family,model in candidates():
        if name not in ['logistic_refined_C0.01','tree_d2_leaf40','forest_d4_leaf25']:continue
        model.fit(train,train.outcome)
        expected=probabilities(model,valid)
        changed=valid.copy();changed['hg']=99;changed['ag']=98;changed['outcome']='D'
        np.testing.assert_array_equal(expected,probabilities(model,changed))
        assert np.allclose(expected.sum(axis=1),1)
