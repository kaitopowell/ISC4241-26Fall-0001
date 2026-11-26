"""Independent numerical audit of saved predictions and the fitted logistic pipeline."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from analysis.models import CLASSES,probabilities,vif_table
from analysis.run import fingerprint
root=Path(__file__).resolve().parents[1];r=root/'reports'
lock=json.loads((r/'selection_lock.json').read_text())
assert fingerprint()==lock['fingerprint']
summary=pd.read_csv(r/'test_metrics.csv').set_index('name')
for name,row in summary.iterrows():
    p=pd.read_csv(r/f'test_predictions_{name}.csv')
    a=p[['p_A','p_D','p_H']].to_numpy()
    assert np.isfinite(a).all() and (a>=0).all()
    np.testing.assert_allclose(a.sum(axis=1),1)
    observed=np.array([list(CLASSES).index(x) for x in p.outcome])
    loss=-np.log(np.clip(a[np.arange(len(p)),observed],np.finfo(float).eps,1)).mean()
    assert np.isclose(loss,row.log_loss)
    assert np.isclose((CLASSES[a.argmax(axis=1)]==p.outcome).mean(),row.accuracy)
bundle=joblib.load(root/'artifacts/development_bundle.joblib')
x=pd.read_csv(root/'data/model_features.csv');x=x[x.season==lock['test_season']]
pipe=bundle['models'][lock['selected_model']]
z=pipe.named_steps['scale'].transform(pipe.named_steps['features'].transform(x))
m=pipe.named_steps['model']
# Avoid BLAS matrix multiplication for an independent check of probability values.
logits=np.einsum('ij,kj->ik',z,m.coef_,optimize=False)+m.intercept_
manual=np.exp(logits-logits.max(axis=1,keepdims=True));manual/=manual.sum(axis=1,keepdims=True)
np.testing.assert_allclose(manual,probabilities(pipe,x),rtol=1e-12,atol=1e-14)
# VIF residual multiplication checked using an explicit elementwise sum as well.
dev=pd.read_csv(root/'data/model_features.csv');dev=dev[dev.season<lock['test_season']]
cols=lock['refined_features'];a=dev[cols].to_numpy()
expected=[]
for j in range(len(cols)):
    y=a[:,j];other=np.column_stack([np.ones(len(a)),np.delete(a,j,axis=1)])
    beta=np.linalg.lstsq(other,y,rcond=None)[0]
    residual=y-np.sum(other*beta[None,:],axis=1)
    expected.append(np.sum((y-y.mean())**2)/np.sum(residual**2))
np.testing.assert_allclose(expected,vif_table(dev[cols]).vif,rtol=1e-12)
print('Verified all saved test metrics, finite normalized probabilities, independent logistic softmax, VIF arithmetic and frozen fingerprint.')
from sklearn.metrics import classification_report
selected_predictions=pd.read_csv(r/f'test_predictions_{lock["selected_model"]}.csv')
report=classification_report(selected_predictions.outcome,selected_predictions.predicted,labels=CLASSES,output_dict=True,zero_division=0)
pd.DataFrame(report).transpose().to_csv(r/'test_classification_report.csv')
from analysis.predict import predict_match
example=predict_match('Arsenal','Chelsea','2024-05-20')
assert np.isclose(sum(example['probabilities'].values()),1)
for home,away,date in [('Arsenal','Chelsea','2026-11-26'),('Arsenal','Arsenal','2024-05-20'),('Unknown club','Chelsea','2024-05-20'),('Arsenal','Chelsea','2022-01-01')]:
    try:predict_match(home,away,date)
    except ValueError:pass
    else:raise AssertionError('Unsupported demonstration request accepted')
print('Historical demo succeeds; stale, in-sample, unknown-team and same-team requests are rejected.')
