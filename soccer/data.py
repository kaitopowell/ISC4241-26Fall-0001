"""Load and validate the eleven Football-Data.co.uk Premier League seasons."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "football_data"

COLUMNS = {
    "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
    "FTHG": "hg", "FTAG": "ag", "FTR": "result",
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast", "HC": "hc", "AC": "ac",
}


def _odds(d: pd.DataFrame, side: str) -> pd.Series:
    """Market-average pre-match odds. Older files call the column BbAv*."""
    for col in (f"Avg{side}", f"BbAv{side}", f"B365{side}"):
        if col in d:
            return d[col]
    raise KeyError(side)


def load_matches(data_dir: Path = DATA) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("E0_*.csv")):
        code = path.stem.split("_")[1]
        start = 2000 + int(code[:2])
        raw = pd.read_csv(path, encoding="utf-8-sig").dropna(how="all")
        d = raw[list(COLUMNS)].rename(columns=COLUMNS).copy()
        d["date"] = pd.to_datetime(d["date"], dayfirst=True, format="mixed")
        d["season"] = f"{start}-{start + 1}"
        # Odds are kept ONLY for the bookmaker benchmark; they are never model inputs.
        for side in "HDA":
            d[f"odds_{side}"] = _odds(raw, side).to_numpy()
        _validate(d, path.name)
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["match_id"] = m["season"] + "_" + m["home"] + "_" + m["away"]
    return m.sort_values(["date", "match_id"]).reset_index(drop=True)


def _validate(d: pd.DataFrame, name: str) -> None:
    assert len(d) == 380, (name, len(d))
    assert d.home.nunique() == 20 and d.away.nunique() == 20, name
    assert not d.duplicated(["home", "away"]).any(), name
    assert d[["date", "home", "away", "hg", "ag", "result"]].notna().all().all(), name
    implied = np.where(d.hg > d.ag, "H", np.where(d.hg < d.ag, "A", "D"))
    assert (implied == d.result).all(), name
