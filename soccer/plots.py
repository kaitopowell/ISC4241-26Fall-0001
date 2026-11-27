"""Diagnostic and results figures (static PNGs for the notebook and report)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
FAMILY = {"baseline": GRAY, "baseline logistic": GRAY, "logistic": BLUE, "tree": ORANGE, "forest": AQUA}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#ecebe7", "grid.linewidth": 0.8, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.axisbelow": True})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def season_rates(f, path):
    r = f.groupby("season").result.value_counts(normalize=True).unstack()[["H", "D", "A"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(r))
    for col, c, lab in [("H", BLUE, "Home win"), ("D", GRAY, "Draw"), ("A", ORANGE, "Away win")]:
        ax.plot(x, r[col], color=c, lw=2, marker="o", ms=5, label=lab)
    ax.axvspan(list(r.index).index("2020-2021") - .5, list(r.index).index("2020-2021") + .5,
               color="#f3f2ee", zorder=0)
    ax.annotate("No crowds\n(COVID-19)", (list(r.index).index("2020-2021"), r.loc["2020-2021", "H"]),
                xytext=(0, -38), textcoords="offset points", ha="center", color=MUTED, fontsize=8)
    ax.set_xticks(x, [s[2:4] + "/" + s[7:] for s in r.index])
    ax.set_ylabel("Share of matches"); ax.set_ylim(0.15, 0.55)
    ax.set_title("Outcome rates by season - home advantage shrank without crowds")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _save(fig, path)


def correlation(X, path):
    c = X.corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(c)), c.columns, fontsize=8)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            if abs(c.iloc[i, j]) >= .7 and i != j:
                ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(c.iloc[i, j]) > .85 else INK)
    fig.colorbar(im, ax=ax, shrink=.75, label="Pearson r")
    ax.set_title("Correlation of candidate predictors (|r| >= 0.7 labelled)")
    _save(fig, path)


def vif_bars(full, kept, path, final=None):
    panels = [(full, f"All {len(full)} candidate predictors"),
              (kept, f"Classic pruning ({len(kept)} kept, Elo lost)")]
    if final is not None:
        panels.append((final, f"Selected model ({len(final)} kept, Elo kept)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6),
                             gridspec_kw={"width_ratios": [1.3] + [1] * (len(panels) - 1)})
    for ax, (v, title) in zip(axes, panels):
        v = v.replace(np.inf, 1e3).sort_values()
        ax.barh(v.index, np.minimum(v, 60), color=[ORANGE if x > 5 else BLUE for x in v], height=.65)
        ax.axvline(5, color=INK, lw=1, ls="--")
        ax.set_xlabel("VIF (capped at 60; dashed = 5)"); ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8); ax.grid(axis="y", visible=False)
        for i, x in enumerate(v):
            ax.text(min(x, 60), i, " inf" if x >= 1e3 else f" {x:.1f}", va="center", fontsize=7, color=MUTED)
    _save(fig, path)


def _binned(ax, x, resid, bins=20, xlabel=""):
    d = pd.DataFrame({"x": x, "r": resid})
    d["bin"] = pd.qcut(d.x, bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(x=("x", "mean"), r=("r", "mean"), n=("r", "size"),
                                            sd=("r", "std"))
    se = 2 * g.sd / np.sqrt(g.n)
    ax.fill_between(g.x, -se, se, color="#ecebe7", step=None, label="±2 SE")
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(g.x, g.r, "o", color=BLUE, ms=5)
    out = (g.r.abs() > se).sum()
    ax.set_xlabel(xlabel); ax.set_ylabel("Mean residual (y - p)")
    ax.set_title(f"Binned residuals vs {xlabel} ({out}/{len(g)} bins outside)")


def logistic_diagnostics(dev, kept, path, influence_csv):
    """Lecture 8/9 style diagnostics on a statsmodels Logit fit of the pruned model."""
    X = (dev[kept] - dev[kept].mean()) / dev[kept].std()
    res = sm.GLM(dev.home_win, sm.add_constant(X), family=sm.families.Binomial()).fit()
    p = res.fittedvalues.to_numpy()
    infl = res.get_influence()
    lev, cook = infl.hat_matrix_diag, infl.cooks_distance[0]
    pres = res.resid_pearson / np.sqrt(1 - lev)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _binned(axes[0, 0], p, dev.home_win - p, xlabel="fitted probability")
    _binned(axes[0, 1], dev.elo_diff, dev.home_win - p, xlabel="elo_diff")
    second = "gd20_diff" if "gd20_diff" in kept else kept[1]
    _binned(axes[1, 0], dev[second], dev.home_win - p, xlabel=second)
    ax = axes[1, 1]
    ax.scatter(lev, pres, s=4 + 4000 * cook, color=BLUE, alpha=.35, edgecolor="white", lw=.3)
    thr = 2 * (len(kept) + 1) / len(dev)
    ax.axvline(thr, color=ORANGE, ls="--", lw=1); ax.text(thr, ax.get_ylim()[1] * .9, " 2(p+1)/n", color=MUTED, fontsize=8)
    ax.set_xlabel("Leverage h_ii"); ax.set_ylabel("Studentised Pearson residual")
    ax.set_title(f"Influence (bubble = Cook's D; max {cook.max():.4f})")
    _save(fig, path)
    top = dev.assign(leverage=lev, cooks_d=cook, fitted=p, std_resid=pres).nlargest(10, "cooks_d")
    top[["match_id", "date", "result", "fitted", "leverage", "cooks_d", "std_resid"] + kept].to_csv(influence_csv, index=False)
    (path.parent.parent / "logit_statsmodels_summary.txt").write_text(str(res.summary()))
    return res


def residuals_by_season(dev, oof, path):
    d = oof.merge(dev[["match_id", "season", "no_crowd"]], on="match_id")
    d["r"] = d.y - d.p
    g = d.groupby("season").r.agg(["mean", "std", "size"])
    se = 2 * g["std"] / np.sqrt(g["size"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.errorbar(range(len(g)), g["mean"], yerr=se, fmt="o", color=BLUE, ecolor="#c9c8c2", capsize=3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(g)), [s[2:4] + "/" + s[7:] for s in g.index])
    ax.set_ylabel("Mean OOF residual (y - p)")
    ax.set_title("Out-of-fold residual by season (systematic over/under-prediction?)")
    _save(fig, path)


def tree_depth_curve(folds, path):
    f = folds[folds.model.str.startswith(("tree", "forest"))].copy()
    g = f.groupby("model")[["accuracy", "train_accuracy"]].mean()
    g = g.sort_values("train_accuracy")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(g))
    ax.plot(x, g.train_accuracy, "o-", color=ORANGE, lw=2, label="Training accuracy")
    ax.plot(x, g.accuracy, "o-", color=BLUE, lw=2, label="Validation accuracy (10-fold mean)")
    ax.set_xticks(x, g.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy"); ax.legend(frameon=False)
    ax.set_title("Bias-variance: deeper trees fit training data better but do not validate better")
    _save(fig, path)


def cv_comparison(summary, book, path, selected=None):
    s = summary.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(9, max(4, .27 * len(s))))
    ax.barh(s.model, s.accuracy - .5, left=.5, xerr=s.fold_acc_sd / np.sqrt(10), height=.62,
            color=[FAMILY[f] for f in s.family], error_kw={"ecolor": "#9c9b96", "lw": 1})
    ax.grid(axis="y", visible=False)
    ax.axvline(book["accuracy"], color=INK, ls="--", lw=1)
    ax.text(book["accuracy"] - .001, len(s) - .4, f"Bookmaker {book['accuracy']:.1%} ", ha="right",
            fontsize=8, color=INK)
    for i, (m, a) in enumerate(zip(s.model, s.accuracy)):
        ax.text(.5015, i, f"{a:.1%}", va="center", fontsize=7, color="white")
    ax.set_xlim(.5, .7); ax.set_xlabel("Pooled out-of-fold accuracy (whiskers = ± 1 SE across folds)")
    ax.tick_params(axis="y", labelsize=7.5)
    if selected:
        for t in ax.get_yticklabels():
            if t.get_text() == selected:
                t.set_fontweight("bold"); t.set_text(selected + "  (selected)")
        ax.set_yticks(range(len(s)), [m + "  << selected" if m == selected else m for m in s.model])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [GRAY, BLUE, ORANGE, AQUA]]
    ax.legend(handles, ["Baselines", "Logistic", "Decision tree", "Random forest"], frameon=True,
              facecolor="white", edgecolor="white", loc="lower right")
    ax.set_title("10-fold chronological CV, 2016-17 to 2023-24 (home win vs not)")
    _save(fig, path)


def fold_lines(folds, models, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for m, c in zip(dict.fromkeys(models), [GRAY, BLUE, AQUA, ORANGE]):
        d = folds[folds.model == m]
        ax.plot(d.fold, d.accuracy, "o-", color=c, lw=2, ms=5, label=m)
    ax.set_xticks(range(1, 11)); ax.set_xlabel("Fold (later folds = later seasons)"); ax.set_ylabel("Validation accuracy")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Accuracy in each of the 10 validation folds")
    _save(fig, path)


def oof_calibration_roc(oofs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for (name, o), c in zip(oofs.items(), [GRAY, BLUE, AQUA]):
        pt, pp = calibration_curve(o.y, o.p, n_bins=10, strategy="quantile")
        axes[0].plot(pp, pt, "o-", color=c, lw=2, ms=4, label=name)
        fpr, tpr, _ = roc_curve(o.y, o.p)
        axes[1].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(o.y, o.p):.3f})")
    for ax in axes:
        ax.plot([0, 1], [0, 1], color="#c9c8c2", ls="--", lw=1)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set(xlabel="Predicted P(home win)", ylabel="Observed home-win rate", title="Calibration (out-of-fold)")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)", title="ROC (out-of-fold)")
    _save(fig, path)


def coefficients(pipe, path, csv):
    model = pipe.named_steps["model"]
    cols = pipe.named_steps["cols"]
    names = cols.kept_ + list(cols.extra)
    if not hasattr(model, "coef_"):
        imp = pd.Series(model.feature_importances_, index=names).sort_values()
        imp.rename("importance").to_csv(csv)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(imp.index, imp, color=AQUA)
        ax.set_title("Feature importance"); _save(fig, path); return
    coef = pd.Series(model.coef_[0], index=names).sort_values()
    pd.DataFrame({"std_coef": coef, "odds_ratio_per_sd": np.exp(coef)}).to_csv(csv)
    fig, ax = plt.subplots(figsize=(7, .35 * len(coef) + 1.5))
    ax.barh(coef.index, coef, color=[BLUE if v > 0 else ORANGE for v in coef], height=.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Standardised coefficient (log-odds per 1 SD)")
    ax.set_title("Selected model: what pushes P(home win) up or down")
    _save(fig, path)


def test_panel(preds, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cm = confusion_matrix(preds.home_win, preds.p_selected >= .5)
    ax = axes[0]; ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["Pred: not home win", "Pred: home win"]); ax.set_yticks([0, 1], ["Actual: not", "Actual: home win"])
    ax.set_title(f"Selected model on 2024-25 (accuracy {np.trace(cm) / cm.sum():.1%})")
    for col, c, lab in [("p_logit_baseline_form5", GRAY, "Baseline logistic"), ("p_selected", BLUE, "Selected"),
                        ("p_bookmaker", INK, "Bookmaker")]:
        fpr, tpr, _ = roc_curve(preds.home_win, preds[col])
        axes[1].plot(fpr, tpr, color=c, lw=2, ls="--" if c == INK else "-",
                     label=f"{lab} (AUC {roc_auc_score(preds.home_win, preds[col]):.3f})")
    axes[1].plot([0, 1], [0, 1], color="#c9c8c2", ls=":")
    axes[1].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC on the held-out 2024-25 season")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, path)
