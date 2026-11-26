import json

import numpy as np
import pandas as pd
import pytest

from soccer.data import ROOT, load_matches
from soccer.features import ALL_FEATURES, build_features
from soccer.models import Columns, chrono_folds, initial_candidates, vif_table
from soccer.refine import refined_candidates


@pytest.fixture(scope="module")
def matches():
    return load_matches()


@pytest.fixture(scope="module")
def feats(matches):
    return build_features(matches)[0]


def test_eleven_complete_seasons(matches):
    assert matches.season.nunique() == 11
    assert matches.groupby("season").size().eq(380).all()


def _features_for(matches, match_id):
    f, _ = build_features(matches)
    return f.set_index("match_id").loc[match_id, ALL_FEATURES].astype(float)


def test_current_match_result_and_shots_do_not_leak(matches, feats):
    target = feats.match_id.iloc[1500]
    changed = matches.copy()
    i = changed.index[changed.match_id == target][0]
    changed.loc[i, ["hg", "ag", "hs", "as_", "hst", "ast"]] = [9, 0, 40, 0, 30, 0]
    changed.loc[i, "result"] = "H"
    pd.testing.assert_series_equal(_features_for(matches, target), _features_for(changed, target))


def test_future_and_same_day_matches_do_not_leak(matches, feats):
    target = feats.iloc[2000]
    changed = matches.copy()
    later = changed.date >= target.date
    later &= changed.match_id != target.match_id
    changed.loc[later, ["hg", "ag", "hs", "as_", "hst", "ast"]] = 0
    changed.loc[later, "result"] = "D"
    pd.testing.assert_series_equal(_features_for(matches, target.match_id),
                                   _features_for(changed, target.match_id))


def test_odds_are_never_model_inputs():
    for _, pipe in {**initial_candidates(), **refined_candidates()}.values():
        cols = pipe.named_steps["cols"].columns
        assert not any("odds" in c for c in cols)


def test_folds_are_chronological_and_date_aligned(feats):
    dev = feats[(feats.season >= "2016-2017") & (feats.season <= "2023-2024")].reset_index(drop=True)
    folds = list(chrono_folds(dev.date))
    assert len(folds) == 10
    for _, tr, va in folds:
        assert dev.date.iloc[tr].max() < dev.date.iloc[va].min()


def test_exact_linear_combination_has_infinite_vif(feats):
    v = vif_table(feats[["gf20_diff", "ga20_diff", "gd20_diff", "elo_diff"]])
    assert np.isinf(v["gd20_diff"])


def test_target_aware_pruning_keeps_elo_classic_does_not(feats):
    dev = feats[feats.season <= "2023-2024"]
    classic = Columns(ALL_FEATURES, prune=True).fit(dev, dev.home_win)
    target = Columns(ALL_FEATURES, prune="target").fit(dev, dev.home_win)
    assert "elo_diff" not in classic.kept_ and "elo_diff" in target.kept_
    assert vif_table(dev[target.kept_]).max() <= 5


def test_test_season_untouched_by_selection():
    lock = json.loads((ROOT / "reports" / "selection_lock.json").read_text())
    assert lock["test"] == "2024-2025" and "2024-2025" not in lock["development"]
    assert lock["development"].split("..")[1].strip().startswith("2023-2024")
