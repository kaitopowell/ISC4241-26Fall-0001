"""Fold-local preprocessing and date-grouped forward validation."""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, log_loss, roc_auc_score
from .data import FULL, REDUCED

CLASSES = np.array(['A','D','H'])
LABELS = ['Away win','Draw','Home win']

def vif_table(X):
    """OLS R-squared against all OTHER columns, with an intercept."""
    a = np.asarray(X, dtype=float)
    rows = []
    for j, name in enumerate(X.columns):
        y = a[:,j]
        sst = np.sum((y-y.mean())**2)
        if sst < 1e-12:
            vif = np.inf
        else:
            others = np.column_stack([np.ones(len(a)),np.delete(a,j,axis=1)])
            residual = y-others@np.linalg.lstsq(others,y,rcond=None)[0]
            ratio = np.sum(residual**2)/sst
            vif = np.inf if ratio < 1e-10 else 1/ratio
        rows.append({'feature':name,'vif':vif})
    return pd.DataFrame(rows)

class FeatureSet(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, prune=False, threshold=5.):
        self.columns = columns
        self.prune = prune
        self.threshold = threshold
    def fit(self, X, y=None):
        self.kept_ = list(self.columns)
        self.removed_ = []
        if self.prune:
            while len(self.kept_)>1:
                values = vif_table(X[self.kept_])
                worst = values.loc[values.vif.idxmax()]
                if worst.vif <= self.threshold: break
                self.kept_.remove(worst.feature)
                self.removed_.append({'feature':worst.feature,'vif':float(worst.vif)})
        return self
    def transform(self, X):
        return X[self.kept_].to_numpy(dtype=float)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_, dtype=object)

def make_pipeline(model, columns=REDUCED, prune=True):
    return Pipeline([('features',FeatureSet(columns=columns,prune=prune)),
                     ('scale',StandardScaler()),('model',model)])

def candidates():
    result = []
    def add(name,family,model,columns=REDUCED,prune=True):
        result.append((name,family,make_pipeline(model,columns,prune)))
    add('majority','majority',DummyClassifier(strategy='most_frequent'),prune=False)
    add('prior','prior',DummyClassifier(strategy='prior'),prune=False)
    # Deliberately retain algebraic redundancy as the before-refinement comparison.
    add('logistic_initial','logistic_initial',LogisticRegression(C=1,max_iter=3000),FULL,False)
    for c in [.01,.1,1,10]:
        add(f'logistic_refined_C{c}','logistic_refined',LogisticRegression(C=c,max_iter=3000))
    add('logistic_no_elo','logistic_no_elo',LogisticRegression(C=.1,max_iter=3000),[c for c in REDUCED if c!='elo_diff'])
    for depth in [2,4,6]:
        for leaf in [15,40]:
            add(f'tree_d{depth}_leaf{leaf}','decision_tree',DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=42))
    for depth in [4,8]:
        for leaf in [10,25]:
            add(f'forest_d{depth}_leaf{leaf}','random_forest',RandomForestClassifier(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,max_features='sqrt',random_state=42,n_jobs=1))
    # Tests draw-recall tradeoff without synthetic matches or pre-CV oversampling.
    add('logistic_balanced','logistic_balanced',LogisticRegression(C=.1,class_weight='balanced',max_iter=3000))
    return result

def forward_splits(df, n_splits=10):
    dates = np.array(sorted(df.date.unique()))
    first = max(30,int(len(dates)*.4))
    blocks = np.array_split(dates[first:],n_splits)
    if min(map(len,blocks))<2: raise ValueError('Too few dates for requested folds')
    for fold, block in enumerate(blocks,1):
        train = np.flatnonzero(df.date < block[0])
        valid = np.flatnonzero(df.date.isin(block))
        if len(set(df.iloc[train].outcome))!=3 or len(set(df.iloc[valid].outcome))!=3:
            raise ValueError(f'Fold {fold} lacks one of the three classes; use fewer folds')
        yield fold, train, valid

def probabilities(model, X):
    p = model.predict_proba(X)
    return p[:,[list(model.classes_).index(k) for k in CLASSES]]

def metrics(y, p):
    y = np.asarray(y)
    pred = CLASSES[p.argmax(axis=1)]
    onehot = (y[:,None]==CLASSES).astype(float)
    result = {'accuracy':accuracy_score(y,pred), 'macro_f1':f1_score(y,pred,labels=CLASSES,average='macro',zero_division=0),
              'balanced_accuracy':balanced_accuracy_score(y,pred),
              'log_loss':log_loss(y,p,labels=CLASSES),
              'brier_multiclass':np.mean(np.sum((onehot-p)**2,axis=1)),
              'draw_recall':np.mean(pred[y=='D']=='D') if np.any(y=='D') else np.nan}
    result['auc_macro_ovr'] = roc_auc_score(y,p,labels=CLASSES,multi_class='ovr',average='macro') if len(set(y))==3 else np.nan
    return result
