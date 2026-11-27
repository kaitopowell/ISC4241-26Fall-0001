"""Candidate models, VIF pruning, chronological 10-fold CV and metrics."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import ALL_FEATURES, FORM5

N_FOLDS = 10
VIF_LIMIT = 5.0


# ---------------------------------------------------------------- collinearity
def vif_table(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R^2_j) from regressing column j on all the others (Lecture 8)."""
    sd = X.std(ddof=0)
    const = sd[sd < 1e-12].index          # e.g. no_crowd in folds that end before 2020
    Z = (X.drop(columns=const) - X.mean()) / sd
    Z = Z.drop(columns=const)
    out = {c: 1.0 for c in const}          # a constant column cannot be collinear
    for c in Z:
        others = Z.drop(columns=c).to_numpy()
        A = np.column_stack([np.ones(len(Z)), others])
        beta, *_ = np.linalg.lstsq(A, Z[c].to_numpy(), rcond=None)
        resid = Z[c].to_numpy() - A @ beta
        r2 = 1 - resid.var() / Z[c].var(ddof=0)
        out[c] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return pd.Series(out).sort_values(ascending=False)


class Columns(BaseEstimator, TransformerMixin):
    """Select columns; optionally drop the highest-VIF column until all VIF <= limit.

    Pruning is fitted on the TRAINING fold only, so validation rows never influence
    which predictors are kept.
    """

    def __init__(self, columns, prune=False, limit=VIF_LIMIT, extra=()):
        self.columns, self.prune, self.limit, self.extra = columns, prune, limit, extra

    def fit(self, X, y=None):
        keep = list(self.columns)
        self.history_ = []
        if self.prune:
            # "target": among predictors with VIF > limit, drop the one LEAST correlated
            # with the training-fold outcome (keeps strong-but-collinear Elo).
            # True: classic rule, drop the highest-VIF predictor.
            with np.errstate(invalid="ignore", divide="ignore"):   # constant no_crowd in early folds
                strength = (X[keep].corrwith(pd.Series(np.asarray(y), index=X.index)).abs().fillna(0)
                            if self.prune == "target" else None)
            while len(keep) > 1:
                v = vif_table(X[keep])
                if v.iloc[0] <= self.limit:
                    break
                high = v[v > self.limit].index
                if np.isinf(v.iloc[0]):
                    high = v[np.isinf(v)].index
                drop = strength[high].idxmin() if self.prune == "target" else v.index[0]
                self.history_.append((drop, float(v[drop])))
                keep.remove(drop)
        self.kept_ = keep
        return self

    def transform(self, X):
        Z = X[self.kept_].copy()
        for name in self.extra:            # diagnostic-driven terms, see refine stage
            a, _, b = name.partition("*")
            Z[name] = X[a] * X[b] if b else X[a] ** 2
        return Z


def make(model, columns, prune=False, extra=()):
    return Pipeline([("cols", Columns(columns, prune=prune, extra=extra)),
                     ("scale", StandardScaler()), ("model", model)])


def logit(C=1.0, penalty="l2"):
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000)


# ------------------------------------------------------------------ candidates
def initial_candidates():
    """Stage 1 grid, fixed BEFORE any diagnostics were inspected."""
    c = {"majority": ("baseline", make(DummyClassifier(strategy="most_frequent"), FORM5)),
         "logit_baseline_form5": ("baseline logistic", make(logit(1e4), FORM5)),
         "logit_full_unpenalised": ("logistic", make(logit(1e4), ALL_FEATURES))}
    for C in (0.01, 0.1, 1.0):
        c[f"logit_vif_C{C}"] = ("logistic", make(logit(C), ALL_FEATURES, prune=True))
    for C in (0.01, 0.03, 0.1):
        c[f"logit_lasso_C{C}"] = ("logistic", make(logit(C, "l1"), ALL_FEATURES))
    for depth in (2, 3, 4, 6):
        for leaf in (20, 60):
            c[f"tree_d{depth}_leaf{leaf}"] = ("tree", make(
                DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0), ALL_FEATURES))
    for depth in (4, 8, None):
        for leaf in (20, 50):
            c[f"forest_d{depth}_leaf{leaf}"] = ("forest", make(RandomForestClassifier(
                n_estimators=300, max_depth=depth, min_samples_leaf=leaf, max_features=0.5,
                random_state=0, n_jobs=-1), ALL_FEATURES))
    return c


# ---------------------------------------------------------- chronological CV
def chrono_folds(dates: pd.Series, k: int = N_FOLDS):
    """Expanding-window folds: 12 date-aligned blocks; fold i trains on blocks
    1..i+1 and validates on block i+2. Same-day matches never straddle a cut."""
    n = len(dates)
    cuts = [0]
    for b in np.linspace(0, n, k + 3).astype(int)[1:-1]:
        while 0 < b < n and dates.iloc[b] == dates.iloc[b - 1]:
            b += 1
        cuts.append(b)
    cuts.append(n)
    for i in range(k):
        yield i + 1, np.arange(0, cuts[i + 2]), np.arange(cuts[i + 2], cuts[i + 3])


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y)
    pred = (p >= threshold).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    return {"accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
            "sensitivity": tp / max((y == 1).sum(), 1),
            "specificity": tn / max((y == 0).sum(), 1)}


def cross_validate(pipe, dev: pd.DataFrame, target="home_win"):
    """Return (per-fold metrics, out-of-fold predictions)."""
    folds, oof = [], []
    for fold, tr, va in chrono_folds(dev.date):
        m = clone(pipe).fit(dev.iloc[tr], dev[target].iloc[tr])
        p = m.predict_proba(dev.iloc[va])[:, 1]
        p_tr = m.predict_proba(dev.iloc[tr])[:, 1]
        folds.append({"fold": fold, "n_train": len(tr), "n_valid": len(va),
                      **binary_metrics(dev[target].iloc[va], p),
                      "train_accuracy": accuracy_score(dev[target].iloc[tr], p_tr >= 0.5)})
        oof.append(pd.DataFrame({"match_id": dev.match_id.iloc[va].to_numpy(), "fold": fold,
                                 "y": dev[target].iloc[va].to_numpy(), "p": p}))
    return pd.DataFrame(folds), pd.concat(oof, ignore_index=True)


def bookmaker_prob(d: pd.DataFrame) -> pd.DataFrame:
    """Overround-normalised implied probabilities (benchmark only)."""
    inv = 1 / d[["odds_H", "odds_D", "odds_A"]]
    q = inv.div(inv.sum(axis=1), axis=0)
    q.columns = ["H", "D", "A"]
    return q
