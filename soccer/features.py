"""Pre-match features built strictly from matches played on EARLIER dates.

All matches on a calendar date are predicted first; only then are that day's
results added to team histories and Elo ratings. A team's history therefore can
never contain the match being predicted or any same-day/future match.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

ELO_START, ELO_K, ELO_HOME, ELO_CARRY = 1500.0, 20.0, 60.0, 0.8
PROMOTED_ELO = 1420.0          # promoted clubs are weaker than the league average
MIN_GAMES = 5                  # warm-up: each team needs 5 recent league games
MAX_AGE_DAYS = 400             # ignore stale history (e.g. clubs returning after years away)

# Matches played without (or with almost no) spectators because of COVID-19.
NO_CROWD = (pd.Timestamp("2020-06-17"), pd.Timestamp("2021-05-17"))

FORM5 = ["ppg5_diff", "gf5_diff", "ga5_diff"]                  # original project recipe
LONG = ["ppg20_diff", "gf20_diff", "ga20_diff", "gd20_diff"]
SHOTS = ["shots20_diff", "shots_against20_diff", "sot20_diff", "sot_against20_diff", "sot_share20_diff"]
VENUE = ["venue_ppg_diff", "venue_gd_diff"]
OTHER = ["elo_diff", "rest_diff", "no_crowd"]
ALL_FEATURES = OTHER[:1] + FORM5 + LONG + SHOTS + VENUE + OTHER[1:]


def _goal_multiplier(margin: int) -> float:
    margin = abs(margin)
    return 1.0 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8


class _State:
    def __init__(self):
        self.games = defaultdict(list)   # team -> list of dicts (one per played match)
        self.elo = {}
        self.season = None

    def new_season(self, season, teams):
        # Regress carried-over ratings toward the mean; promoted clubs start low.
        self.elo = {t: ELO_START + ELO_CARRY * (r - ELO_START) for t, r in self.elo.items()}
        for t in teams:
            self.elo.setdefault(t, PROMOTED_ELO)
        self.season = season

    def recent(self, team, date, n, venue=None):
        g = [x for x in self.games[team] if (date - x["date"]).days <= MAX_AGE_DAYS]
        if venue:
            g = [x for x in g if x["venue"] == venue]
        return g[-n:]

    def team_features(self, team, date, venue):
        last20 = self.recent(team, date, 20)
        if len(last20) < MIN_GAMES:
            return None
        last5 = last20[-5:]
        mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
        f = {
            "ppg5": mean(last5, "pts"), "gf5": mean(last5, "gf"), "ga5": mean(last5, "ga"),
            "ppg20": mean(last20, "pts"), "gf20": mean(last20, "gf"), "ga20": mean(last20, "ga"),
            "shots20": mean(last20, "sh"), "shots_against20": mean(last20, "sh_a"),
            "sot20": mean(last20, "sot"), "sot_against20": mean(last20, "sot_a"),
        }
        f["gd20"] = f["gf20"] - f["ga20"]
        tot = f["sot20"] + f["sot_against20"]
        f["sot_share20"] = f["sot20"] / tot if tot else 0.5
        ven = self.recent(team, date, 10, venue)
        f["venue_ppg"] = mean(ven, "pts") if ven else f["ppg20"]
        f["venue_gd"] = (mean(ven, "gf") - mean(ven, "ga")) if ven else f["gd20"]
        f["rest"] = min((date - last20[-1]["date"]).days, 14)
        return f

    def update(self, r):
        hg, ag = int(r.hg), int(r.ag)
        hp, ap = (3, 0) if hg > ag else (0, 3) if hg < ag else (1, 1)
        common = dict(date=r.date)
        self.games[r.home].append({**common, "venue": "H", "gf": hg, "ga": ag, "pts": hp,
                                   "sh": r.hs, "sh_a": r.as_, "sot": r.hst, "sot_a": r.ast})
        self.games[r.away].append({**common, "venue": "A", "gf": ag, "ga": hg, "pts": ap,
                                   "sh": r.as_, "sh_a": r.hs, "sot": r.ast, "sot_a": r.hst})
        # Elo with home advantage and goal-margin multiplier.
        eh, ea = self.elo[r.home], self.elo[r.away]
        expected = 1 / (1 + 10 ** (-(eh + ELO_HOME - ea) / 400))
        actual = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
        delta = ELO_K * _goal_multiplier(hg - ag) * (actual - expected)
        self.elo[r.home], self.elo[r.away] = eh + delta, ea - delta


def build_features(matches: pd.DataFrame):
    """Return (feature table, excluded warm-up rows)."""
    state, rows, excluded = _State(), [], []
    for date, day in matches.groupby("date", sort=True):
        season = day.season.iloc[0]
        if season != state.season:
            teams = matches.loc[matches.season == season, "home"].unique()
            state.new_season(season, teams)
        for r in day.itertuples(index=False):
            fh = state.team_features(r.home, date, "H")
            fa = state.team_features(r.away, date, "A")
            if fh is None or fa is None:
                excluded.append({"match_id": r.match_id, "season": season, "date": date})
                continue
            row = r._asdict()
            row.update({f"{k}_diff": fh[k] - fa[k] for k in fh})
            row["elo_diff"] = state.elo[r.home] - state.elo[r.away]
            row["no_crowd"] = int(NO_CROWD[0] <= date <= NO_CROWD[1])
            rows.append(row)
        for r in day.itertuples(index=False):       # results enter history AFTER the day
            state.update(r)
    f = pd.DataFrame(rows)
    f["home_win"] = (f.result == "H").astype(int)
    return f, pd.DataFrame(excluded)
