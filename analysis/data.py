"""Recover the team's cached API results without executing its notebook."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ANOTHERBEST-proj4_comb_methods - BESTANOTHER.ipynb'
RENAME = {'idEvent':'match_id', 'idLeague':'league_id', 'strSeason':'season',
          'dateEvent':'date', 'idHomeTeam':'home_id', 'idAwayTeam':'away_id',
          'strHomeTeam':'home', 'strAwayTeam':'away', 'intHomeScore':'hg',
          'intAwayScore':'ag', 'strStatus':'status'}

def extract_cached():
    notebook = json.loads(SOURCE.read_text())
    records = []
    cells = []
    for i, cell in enumerate(notebook['cells']):
        code = ''.join(cell.get('source', []))
        if 'eventsseason.php' not in code or 'id=4328' not in code:
            continue
        text = ''.join(''.join(o.get('text', [])) for o in cell.get('outputs', [])
                       if o.get('output_type') == 'stream')
        payload = ast.literal_eval(text)  # literals only, never eval/exec
        if not isinstance(payload, list):
            raise ValueError('Expected complete saved event list')
        records.extend({new: r.get(old) for old, new in RENAME.items()} for r in payload)
        cells.append(i)
    if not records:
        raise ValueError('No cached Premier League records found')
    frame = pd.DataFrame(records)
    path = ROOT / 'data/raw/cached_fixtures.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    provenance = {'source_notebook': SOURCE.name, 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  'source_cells_zero_based': cells, 'rows': len(frame),
                  'source': 'TheSportsDB league 4328, saved notebook stream outputs',
                  'original_retrieval_time': 'Not recorded; 2024-25 snapshot has incomplete results. Not refreshed.',
                  'extraction': 'Selected event metadata and scores; original notebook never executed.'}
    (ROOT / 'data/raw/provenance.json').write_text(json.dumps(provenance, indent=2))
    return frame

def clean_fixtures(raw):
    df = raw.copy()
    audit = [{'step':'cached records', 'remaining':len(df), 'removed':0}]
    def retain(mask, reason):
        nonlocal df
        before = len(df)
        df = df.loc[mask].copy()
        audit.append({'step':reason, 'remaining':len(df), 'removed':before-len(df)})
    for c in ['match_id', 'league_id', 'home_id', 'away_id']:
        df[c] = df[c].astype('string')
    retain(df.league_id == '4328', 'Premier League only')
    for _, g in df.groupby('match_id'):
        if len(g) > 1 and len(g.drop_duplicates()) > 1:
            raise ValueError('Conflicting duplicate match ID')
    retain(~df.duplicated('match_id'), 'remove identical repeated match IDs')
    df['date'] = pd.to_datetime(df.date, errors='coerce').dt.normalize()
    for c in ['hg','ag']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    retain(df.status == 'Match Finished', 'completed matches only')
    retain(df[['date','home_id','away_id','home','away','hg','ag']].notna().all(axis=1), 'valid dates, teams and scores')
    retain((df.home_id != df.away_id) & (df.hg >= 0) & (df.ag >= 0) &
           (df.hg % 1 == 0) & (df.ag % 1 == 0), 'valid opponents and integer nonnegative scores')
    df[['hg','ag']] = df[['hg','ag']].astype(int)
    if df.duplicated(['season','home_id','away_id']).any():
        raise ValueError('Duplicate fixture pairing within a season')
    coverage = []
    complete = []
    for season, g in df.groupby('season'):
        teams = set(g.home_id) | set(g.away_id)
        home_counts = g.home_id.value_counts().reindex(sorted(teams), fill_value=0)
        away_counts = g.away_id.value_counts().reindex(sorted(teams), fill_value=0)
        ok = len(g)==380 and len(teams)==20 and home_counts.eq(19).all() and away_counts.eq(19).all()
        coverage.append({'season':season, 'completed':len(g), 'teams':len(teams), 'complete_double_round_robin':bool(ok)})
        if ok: complete.append(season)
    retain(df.season.isin(complete), 'complete 380-match seasons only')
    # No score-based deletions: unusual but valid football results are informative.
    df = df.sort_values(['date','match_id']).reset_index(drop=True)
    df['outcome'] = np.where(df.hg > df.ag, 'H', np.where(df.hg < df.ag, 'A', 'D'))
    return df, pd.DataFrame(audit), pd.DataFrame(coverage)

FULL = [f'{side}_{stat}' for side in ['home','away'] for stat in ['ppg5','gf5','ga5','gd5','win5','rest']] + ['elo_diff']
REDUCED = ['ppg_diff','gf_diff','ga_diff','win_diff','rest_diff','elo_diff']

class History:
    """All matches on one date are predicted before any result on that date is added."""
    def __init__(self):
        self.games = {}
        self.elo = {}
    def features(self, home, away, date):
        out = {}
        for side, team in [('home',home),('away',away)]:
            past = self.games.get(team, [])
            if len(past)<5: return None
            recent = past[-5:]
            out[f'{side}_ppg5'] = np.mean([r[1] for r in recent])
            out[f'{side}_gf5'] = np.mean([r[2] for r in recent])
            out[f'{side}_ga5'] = np.mean([r[3] for r in recent])
            out[f'{side}_gd5'] = out[f'{side}_gf5'] - out[f'{side}_ga5']
            out[f'{side}_win5'] = np.mean([r[1]==3 for r in recent])
            # Cap at two weeks so summer breaks do not dominate the rest variable.
            out[f'{side}_rest'] = min((pd.Timestamp(date)-recent[-1][0]).days, 14)
        for short, stat in [('ppg','ppg5'),('gf','gf5'),('ga','ga5'),('win','win5'),('rest','rest')]:
            out[f'{short}_diff'] = out[f'home_{stat}']-out[f'away_{stat}']
        out['elo_diff'] = self.elo.get(home,1500)-self.elo.get(away,1500)
        return out
    def update_day(self, day):
        changes = {}
        for r in day.itertuples():
            h,a = r.home_id,r.away_id
            score = 1. if r.hg > r.ag else (0. if r.hg < r.ag else .5)
            expected = 1/(1+10**((self.elo.get(a,1500)-self.elo.get(h,1500)-60)/400))
            delta = 20*(score-expected)
            changes[h] = changes.get(h,0)+delta
            changes[a] = changes.get(a,0)-delta
            hp,ap = ((3,0) if score==1 else ((0,3) if score==0 else (1,1)))
            self.games.setdefault(h,[]).append((r.date,hp,r.hg,r.ag))
            self.games.setdefault(a,[]).append((r.date,ap,r.ag,r.hg))
        for team,change in changes.items(): self.elo[team]=self.elo.get(team,1500)+change

def build_features(fixtures):
    history = History()
    rows = []
    exclusions = []
    for date, day in fixtures.groupby('date', sort=True):
        for r in day.itertuples():
            f = history.features(r.home_id,r.away_id,date)
            if f is None:
                exclusions.append({'match_id':r.match_id,'season':r.season,'reason':'fewer than five recorded prior matches for either team'})
                continue
            rows.append({**r._asdict(), **f})
        history.update_day(day)
    return pd.DataFrame(rows).drop(columns='Index',errors='ignore'), pd.DataFrame(exclusions), history
