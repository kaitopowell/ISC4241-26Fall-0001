import numpy as np
import pandas as pd
from experiments.draw_refinement.run import dataset,labels,score,choose,candidates
from analysis.models import forward_splits,probabilities

def test_decision_rule_changes_labels_without_changing_probabilities():
    p=np.array([[.36,.30,.34],[.10,.20,.70]])
    original=p.copy()
    assert list(labels(p))==['A','H']
    assert list(labels(p,1.5))==['D','H']
    np.testing.assert_array_equal(p,original)
    a=score(['D','H'],p);b=score(['D','H'],p,1.5)
    assert a['log_loss']==b['log_loss'] and a['brier_multiclass']==b['brier_multiclass']
    assert b['draw_recall']==1

def test_nested_boundaries_and_new_holdout():
    dev,test,excluded=dataset()
    assert dev.date.max()<test.date.min()
    assert set(test.season)=={'2024-2025'}
    assert len(test)>150
    assert not set(dev.match_id)&set(test.match_id)
    for _,tr,va in forward_splits(dev,10):
        train=dev.iloc[tr].reset_index(drop=True)
        for _,itr,iva in forward_splits(train,3):
            assert train.iloc[itr].date.max()<train.iloc[iva].date.min()
            assert train.iloc[iva].date.max()<dev.iloc[va].date.min()

def test_selection_enforces_tradeoff_and_reports_fallback():
    rows=pd.DataFrame([{'name':'bad_accuracy','multiplier':1.,'accuracy':.4,'draw_recall':.5,'macro_f1':.6,'log_loss':1.},
                       {'name':'eligible','multiplier':1.5,'accuracy':.55,'draw_recall':.2,'macro_f1':.45,'log_loss':1.},
                       {'name':'no_draws','multiplier':1.,'accuracy':.6,'draw_recall':0.,'macro_f1':.5,'log_loss':.9}])
    winner,table,fallback=choose(rows,.56)
    assert winner['name']=='eligible' and not fallback
    _,_,fallback=choose(rows,.8);assert fallback

def test_symmetric_features_only_depend_on_prematch_inputs():
    dev,_,_=dataset()
    np.testing.assert_allclose(dev.abs_elo_gap,abs(dev.elo_diff))
    np.testing.assert_allclose(dev.total_gf,dev.home_gf5+dev.away_gf5)
    for pipe in candidates().values():
        assert not set(pipe.named_steps['features'].columns)&{'hg','ag','outcome','date','season','match_id'}
