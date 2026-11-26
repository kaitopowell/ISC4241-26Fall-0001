"""Historical-snapshot demo, intentionally refuses unsupported dates or cold-start teams."""
import argparse
import json
import joblib
import pandas as pd
from .data import ROOT, History
from .models import probabilities, CLASSES, LABELS

def predict_match(home,away,date):
    bundle=joblib.load(ROOT/'artifacts/development_bundle.joblib')
    matches=pd.read_csv(ROOT/'data/completed_matches.csv',dtype={'home_id':str,'away_id':str},parse_dates=['date'])
    date=pd.Timestamp(date).normalize()
    trained_until=matches[matches.season<bundle['lock']['test_season']].date.max()
    if date<=trained_until:raise ValueError('Date must be after the model training period; otherwise this would be an in-sample demo.')
    if date>matches.date.max()+pd.Timedelta(days=14):raise ValueError('This archived snapshot is too old for that date. Import verified newer matches and retrain before current forecasts.')
    mapping={str(r.home).casefold():str(r.home_id) for r in matches.itertuples()}
    if home.casefold() not in mapping or away.casefold() not in mapping:raise ValueError('Unknown team. Use names from data/completed_matches.csv.')
    h,a=mapping[home.casefold()],mapping[away.casefold()]
    if h==a:raise ValueError('Home and away must be different teams')
    history=History()
    for _,day in matches[matches.date<date].groupby('date',sort=True):history.update_day(day)
    features=history.features(h,a,date)
    if features is None:raise ValueError('Both teams need at least five recorded previous matches')
    for team in [h,a]:
        if date-history.games[team][-1][0]>pd.Timedelta(days=120):raise ValueError('Team history is stale for this matchup')
    model=bundle['models'][bundle['lock']['selected_model']]
    p=probabilities(model,pd.DataFrame([features]))[0]
    return {'home':home,'away':away,'date':str(date.date()),'model':bundle['lock']['selected_model'],
            'probabilities':dict(zip(LABELS,map(float,p))), 'most_likely':LABELS[int(p.argmax())],
            'notice':'Archived-data demonstration; hypothetical fixture, not a current live prediction. No lineups, injuries or betting odds.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--home',required=True);parser.add_argument('--away',required=True);parser.add_argument('--date',required=True)
    args=parser.parse_args()
    try:print(json.dumps(predict_match(args.home,args.away,args.date),indent=2))
    except ValueError as exc:parser.error(str(exc))
