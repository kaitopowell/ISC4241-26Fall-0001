"""End-to-end pipeline.

    python -m soccer.run --stage develop   # 10-fold CV of the initial grid + diagnostics
    python -m soccer.run --stage refine    # diagnostic-driven candidates, lock the choice
    python -m soccer.run --stage test      # ONE evaluation on the untouched 2024-25 season
"""
import argparse
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import plots
from .data import DATA, ROOT, load_matches
from .features import ALL_FEATURES, build_features
from .models import (binary_metrics, bookmaker_prob, cross_validate, initial_candidates,
                     logit, make, vif_table)
from .refine import refined_candidates

# numpy 2.0 + macOS Accelerate emits spurious matmul warnings (even for random matrices);
# results are finite and verified, see docs/VALIDATION.md.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
OUT = ROOT / "reports"
FIG = OUT / "figures"
ART = ROOT / "artifacts" / "final_model.joblib"
DEV_SEASONS = ("2016-2017", "2023-2024")     # 2014-16 only warm up Elo/form history
TEST_SEASON = "2024-2025"


def load():
    f, excluded = build_features(load_matches())
    dev = f[(f.season >= DEV_SEASONS[0]) & (f.season <= DEV_SEASONS[1])].reset_index(drop=True)
    test = f[f.season == TEST_SEASON].reset_index(drop=True)
    return f, dev, test, excluded


def evaluate_grid(candidates, dev, tag):
    rows, folds, oofs = [], [], {}
    for name, (family, pipe) in candidates.items():
        print(f"  CV {name}", flush=True)
        fm, oof = cross_validate(pipe, dev)
        fm.insert(0, "model", name)
        folds.append(fm)
        oofs[name] = oof
        pooled = binary_metrics(oof.y, oof.p)
        rows.append({"model": name, "family": family, "stage": tag, **pooled,
                     "fold_acc_sd": fm.accuracy.std(), "train_accuracy": fm.train_accuracy.mean()})
    return pd.DataFrame(rows), pd.concat(folds), oofs


def bookmaker_oof(dev, oofs_any):
    ids = oofs_any.match_id
    q = bookmaker_prob(dev.set_index("match_id").loc[ids])
    return binary_metrics(oofs_any.y, q.H.to_numpy())


def develop():
    FIG.mkdir(parents=True, exist_ok=True)
    f, dev, test, excluded = load()
    excluded.to_csv(OUT / "warmup_exclusions.csv", index=False)
    summary, folds, oofs = evaluate_grid(initial_candidates(), dev, "initial")
    book = bookmaker_oof(dev, oofs["majority"])
    summary.to_csv(OUT / "cv_initial.csv", index=False)
    folds.to_csv(OUT / "cv_initial_folds.csv", index=False)
    joblib.dump(oofs, OUT / "oof_initial.joblib")
    json.dump(book, open(OUT / "bookmaker_cv.json", "w"), indent=2)
    # Diagnostics that motivate the refine stage.
    plots.season_rates(f, FIG / "01_home_win_rate_by_season.png")
    plots.correlation(dev[ALL_FEATURES], FIG / "02_correlation_heatmap.png")
    vif_full = vif_table(dev[ALL_FEATURES])
    vif_full.rename("vif").to_csv(OUT / "vif_all_features.csv")
    pruned = clone(initial_candidates()["logit_vif_C0.1"][1]).fit(dev, dev.home_win)
    kept = pruned.named_steps["cols"].kept_
    vif_kept = vif_table(dev[kept])
    vif_kept.rename("vif").to_csv(OUT / "vif_after_pruning.csv")
    plots.vif_bars(vif_full, vif_kept, FIG / "03_vif_before_after.png")
    plots.logistic_diagnostics(dev, kept, FIG / "04_logistic_diagnostics.png", OUT / "influence_top.csv")
    plots.residuals_by_season(dev, oofs["logit_vif_C0.1"], FIG / "05_oof_residual_by_season.png")
    plots.tree_depth_curve(folds, FIG / "06_tree_forest_train_vs_valid.png")
    print(summary.sort_values("accuracy", ascending=False).to_string(index=False))
    print("bookmaker", book, "\nVIF-kept", kept)


def refine():
    f, dev, test, excluded = load()
    init = pd.read_csv(OUT / "cv_initial.csv")
    oofs = joblib.load(OUT / "oof_initial.joblib")
    summary, folds, new_oofs = evaluate_grid(refined_candidates(), dev, "refined")
    oofs.update(new_oofs)
    allsum = pd.concat([init, summary], ignore_index=True)
    allfolds = pd.concat([pd.read_csv(OUT / "cv_initial_folds.csv"), folds])
    allsum = allsum.sort_values(["accuracy", "log_loss"], ascending=[False, True])
    allsum.to_csv(OUT / "cv_all.csv", index=False)
    allfolds.to_csv(OUT / "cv_all_folds.csv", index=False)
    joblib.dump(oofs, OUT / "oof_all.joblib")
    # Selection rule (declared before refine): best pooled OOF accuracy; models within
    # 0.25 points are treated as tied and the lower log loss wins.
    best = allsum.accuracy.max()
    tied = allsum[allsum.accuracy >= best - 0.0025]
    selected = tied.sort_values("log_loss").iloc[0].model
    cands = {**initial_candidates(), **refined_candidates()}
    book = json.load(open(OUT / "bookmaker_cv.json"))
    plots.cv_comparison(allsum, book, FIG / "07_cv_model_comparison.png")
    plots.fold_lines(allfolds, ["logit_baseline_form5", selected, best_of(allsum, "forest"),
                                best_of(allsum, "tree")], FIG / "08_cv_accuracy_by_fold.png")
    plots.oof_calibration_roc({"Baseline logistic (form5)": oofs["logit_baseline_form5"],
                               "Selected": oofs[selected],
                               "Best forest": oofs[best_of(allsum, "forest")]},
                              FIG / "09_oof_calibration_roc.png")
    final = clone(cands[selected][1]).fit(dev, dev.home_win)
    comparators = {n: clone(cands[n][1]).fit(dev, dev.home_win)
                   for n in ["majority", "logit_baseline_form5", best_of(allsum, "forest"),
                             best_of(allsum, "tree")]}
    plots.coefficients(final, FIG / "10_selected_coefficients.png", OUT / "selected_coefficients.csv")
    lock = {"selected": selected, "family": cands[selected][0],
            "rule": "max pooled 10-fold chronological OOF accuracy; ties (<=0.25 pt) -> lower log loss",
            "features": final.named_steps["cols"].kept_ + list(final.named_steps["cols"].extra),
            "development": f"{DEV_SEASONS[0]} .. {DEV_SEASONS[1]} ({len(dev)} matches)",
            "test": TEST_SEASON, "fingerprint": fingerprint()}
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2))
    ART.parent.mkdir(exist_ok=True)
    joblib.dump({"selected": final, **comparators, "_lock": lock}, ART)
    print(allsum.head(15).to_string(index=False))
    print(json.dumps(lock, indent=2))


def best_of(summary, family):
    s = summary[summary.family == family].sort_values(["accuracy", "log_loss"], ascending=[False, True])
    return s.iloc[0].model


def fingerprint():
    files = sorted((ROOT / "soccer").glob("*.py")) + sorted(DATA.glob("E0_*.csv"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in files if p.name != "plots.py")).hexdigest()


def test_stage():
    lock = json.loads((OUT / "selection_lock.json").read_text())
    assert lock["fingerprint"] == fingerprint(), "Code/data changed after the model was locked"
    bundle = joblib.load(ART)
    assert bundle["_lock"] == lock
    f, dev, test, excluded = load()
    rows, preds = [], test[["match_id", "date", "home", "away", "result", "home_win"]].copy()
    for name, m in bundle.items():
        if name.startswith("_"):
            continue
        p = m.predict_proba(test)[:, 1]
        preds[f"p_{name}"] = p
        label = f"{name} ({lock['selected']})" if name == "selected" else name
        rows.append({"model": label, **binary_metrics(test.home_win, p)})
    q = bookmaker_prob(test)
    rows.append({"model": "bookmaker benchmark (not a model input)", **binary_metrics(test.home_win, q.H.to_numpy())})
    preds["p_bookmaker"] = q.H.to_numpy()
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "test_results.csv", index=False)
    preds.to_csv(OUT / "test_predictions.csv", index=False)
    ci = bootstrap(preds, "p_selected", ["p_logit_baseline_form5", "p_majority", "p_bookmaker"])
    (OUT / "test_bootstrap.json").write_text(json.dumps(ci, indent=2))
    plots.test_panel(preds, FIG / "11_test_confusion_roc.png")
    three = three_way(dev, test, lock)
    three.to_csv(OUT / "three_way_test.csv", index=False)
    print(res.to_string(index=False)); print(json.dumps(ci, indent=2)); print(three.to_string(index=False))


def bootstrap(preds, main, others, n=2000, seed=0):
    """Paired bootstrap over match-weeks (blocks of dates within 7 days)."""
    rng = np.random.default_rng(seed)
    week = ((preds.date - preds.date.min()).dt.days // 7).to_numpy()
    blocks = [np.where(week == w)[0] for w in np.unique(week)]
    y = preds.home_win.to_numpy()
    correct = {c: ((preds[c].to_numpy() >= .5).astype(int) == y) for c in [main] + others}
    out = {}
    stats = {c: [] for c in [main] + others}
    diffs = {c: [] for c in others}
    for _ in range(n):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        for c in stats:
            stats[c].append(correct[c][idx].mean())
        for c in others:
            diffs[c].append(correct[main][idx].mean() - correct[c][idx].mean())
    for c in stats:
        out[f"accuracy_{c}"] = [float(np.mean(correct[c])), *np.percentile(stats[c], [2.5, 97.5]).tolist()]
    for c in others:
        out[f"gain_vs_{c}"] = [float(np.mean(correct[main]) - np.mean(correct[c])), *np.percentile(diffs[c], [2.5, 97.5]).tolist()]
    return out


def three_way(dev, test, lock):
    """Secondary task: home/draw/away with the SAME locked feature set (no new tuning)."""
    from sklearn.dummy import DummyClassifier
    from .models import Columns
    sel = joblib.load(ART)["selected"]
    cols = sel.named_steps["cols"]
    C = sel.named_steps["model"].C
    pipe = make(logit(C), cols.columns, prune=cols.prune, extra=cols.extra)
    rows = []
    for name, p in [("majority (home)", make(DummyClassifier(strategy="most_frequent"), ["elo_diff"])),
                    ("multinomial logistic, locked features", pipe)]:
        m = clone(p).fit(dev, dev.result)
        rows.append({"model": name, "accuracy": float((m.predict(test) == test.result).mean()),
                     "predicted_draws": int((m.predict(test) == "D").sum())})
    q = bookmaker_prob(test)
    rows.append({"model": "bookmaker favourite (benchmark)",
                 "accuracy": float((q.idxmax(axis=1) == test.result).mean()),
                 "predicted_draws": int((q.idxmax(axis=1) == "D").sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["develop", "refine", "test"], required=True)
    stage = ap.parse_args().stage
    {"develop": develop, "refine": refine, "test": test_stage}[stage]()
