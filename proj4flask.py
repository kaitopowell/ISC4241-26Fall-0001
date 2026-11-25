from flask import Flask, render_template, jsonify, request, redirect, url_for
import requests
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from IPython.display import display
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKeyConstraint, PrimaryKeyConstraint, ForeignKey, Sequence, Date
import pickle 
import pandas as pd 

app = Flask(__name__)

import os
from dotenv import load_dotenv

load_dotenv() 
database_url = os.getenv("DATABASE_URL")

engine = create_engine(database_url)
base = declarative_base()

class user_info(base): 
    __tablename__ = 'user_info'
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    first_name = Column(String(250))
    last_name = Column(String(250))
    birthday = Column(Date)

class order(base): 
    __tablename__ = 'amount_add'
    
    order_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user_info.id'))
    amount = Column(Float)
    team = Column(String(250))
    win_draw_loss = Column(String(250))

    __table_args__ = (
        {'extend_existing':True},
    )
    
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind = engine)


prem_pred = pickle.load(open("static/prem_pred.pkl", "rb"))
full_teams_data_original = pd.read_csv("static/full_teams_original", index_col= 0)
full_teams_data = pd.read_csv("static/teams_data.csv", index_col= 0)
away_teams = pd.read_csv("static/away_teams_data.csv", index_col= 0).reset_index(drop = True)
home_teams = pd.read_csv("static/home_teams_data.csv", index_col= 0).reset_index(drop = True)

def rmHomeAwayReversalHome(old2):
    new2 = old2.replace(" AVG", " Home AVG")
    return new2

def rmHomeAwayReversalAway(old2):
    new2 = old2.replace(" AVG", " Away AVG")
    return new2

renamer_reversal_home = {"team": "Home",
           "win streak": "win streak home",
           "Hist intScore Home AVG": "Hist intHomeScore AVG"}
renamer_reversal_away =  {"team": "Away",
         "win streak": "win streak away",
         "Hist intScore Away AVG": "Hist intAwayScore AVG"}
def merge(home, away):
    #create data that can be passed into the model
    home_df = pd.DataFrame([home]).rename(columns=rmHomeAwayReversalHome).rename(columns=renamer_reversal_home).reset_index(drop = True)
    away_df = pd.DataFrame([away]).rename(columns=rmHomeAwayReversalAway).rename(columns=renamer_reversal_away).reset_index(drop = True)

    merged_data = pd.merge(home_df, away_df, left_index=True, right_index=True)
    return merged_data


def merge_split(home, away):
    #create data that can be passed into the model
    home_df = pd.DataFrame([home]).reset_index(drop = True)
    away_df = pd.DataFrame([away]).reset_index(drop = True)

    merged_data_split = pd.merge(home_df, away_df, left_index=True, right_index=True)
    return merged_data_split


#############################################################################
# API ROUTES
#############################################################################


@app.route('/api/teams/')
def apiuse():
    teams = full_teams_data["team"].to_dict()
    return jsonify(teams)


@app.route('/api/teams/<id>/')
def apistats(id):
    stats = full_teams_data.loc[int(id)].to_list()
    return jsonify(stats)
    

@app.route('/api/predict/<home>/<away>/')
def predict(home, away):
    # home = request.form.get("home")
    # away =  request.form.get("away")
    home_stats = full_teams_data.loc[int(home)]
    away_stats = full_teams_data.loc[int(away)]
    # stats = full_teams_data.loc[int(id)].to_dict()
    merge_stats = merge(home_stats, away_stats)
    cols_for_prediction = full_teams_data_original.columns.drop(['winner', 'Home result', 'Date'])
    merge_stats_final = merge_stats
    merge_stats_final = merge_stats_final[cols_for_prediction].drop(columns = ["Home","Away"])
    #get prediction from pickle file
    prediction = prem_pred.predict(merge_stats_final)
    return jsonify({"prediction": prediction.tolist(),
                    'merged_stats': merge_stats_final.to_dict(orient='records')})


#############################################################################
# HTML ROUTES
#############################################################################


@app.route('/', methods = ["GET","POST"])
def home():
    teams = full_teams_data["team"].to_dict()
    away_teams_dropdown = away_teams["Away"].to_dict()
    home_teams_dropdown = home_teams["Home"].to_dict()
    home = None
    away =  None
    home_split = None
    away_split = None
    prediction_split = None
    prediction = None
    prediction_label_split = None
    prediction_label = None
    if request.method == "POST":
        home = request.form.get("homeTeam")
        away =  request.form.get("awayTeam")
        home_split = request.form.get("homeTeam_home")
        away_split = request.form.get("awayTeam_away")
        home_stats = full_teams_data.loc[int(home)]
        away_stats = full_teams_data.loc[int(away)]
        home_split_stats = home_teams.loc[int(home_split)]
        away_split_stats = away_teams.loc[int(away_split)]
        # stats = full_teams_data.loc[int(id)].to_dict()
        merge_stats = merge(home_stats, away_stats)
        merge_split_stats = merge_split(home_split_stats, away_split_stats)
        cols_for_prediction = full_teams_data_original.columns.drop(['winner', 'Home result', 'Date'])
        merge_stats_final = merge_stats
        merge_stats_final = merge_stats_final[cols_for_prediction].drop(columns = ["Home","Away"])
        #for split data
        merge_stats_final_split = merge_split_stats
        merge_stats_final_split = merge_stats_final_split[cols_for_prediction].drop(columns = ["Home","Away"])
        #get prediction from pickle file
        prediction = prem_pred.predict(merge_stats_final).tolist()
        if prediction[0] == 1:
            prediction_label = f'{home_stats[0]} Wins!'
        elif prediction[0] == -1:
            prediction_label = f'{away_stats[0]} Wins!'
        else: 
            prediction_label = 'Draw!'
        prediction_split = prem_pred.predict(merge_stats_final_split).tolist()
        if prediction_split[0] == 1:
            prediction_label_split = f'{home_split_stats[0]} Wins!'
        elif prediction_split[0] == -1:
            prediction_label_split = f'{away_split_stats[0]} Wins!'
        else: 
            prediction_label_split = 'Draw!'
        # return jsonify({"prediction": prediction.tolist(),
        #                 'merged_stats': merge_stats_final.to_dict(orient='records')})
    return render_template('index.html', teams = teams, prediction = prediction, prediction_split = prediction_split, home_teams_dropdown = home_teams_dropdown, away_teams_dropdown = away_teams_dropdown, prediction_label = prediction_label, prediction_label_split = prediction_label_split)


@app.route('/Soccer')
def Soccer():
    return render_template('Soccer.html')


@app.route('/Basketball')
def Basketball():
    return render_template('Basketball.html')


@app.route('/Football')
def Football():
    return render_template('Football.html')


@app.route('/Hockey')
def Hockey():
    return render_template('Hockey.html')


@app.route('/submissions')
def submissions():
    return render_template('submissions.html')


@app.route('/sub', methods = ["GET", "POST"])
def sub():
    session = Session()
    if request.method == 'POST':
        first_name = request.form.get('first name')
        last_name = request.form.get('last name')
        birthday = request.form.get('birthday')
        amount = request.form.get('amount')
        team= request.form.get('team')
        win_draw_loss = request.form.get('wdl')

        new_user = user_info(first_name =first_name, last_name = last_name, birthday = birthday)
       
        session.add(new_user)
        session.commit()

        new_order = order(user_id = new_user.id  ,amount = amount,  team = team , win_draw_loss = win_draw_loss)
        session.add(new_order)
        session.commit()

        return redirect(url_for('confirmation', first_name=first_name, last_name=last_name, birthday=birthday, user_id = new_user.id  ,amount = amount,  team = team , win_draw_loss = win_draw_loss))

    return render_template('submissions.html')


@app.route('/confirmation')
def confirmation():
    first_name = request.args.get('first_name')
    last_name = request.args.get('last_name')
    birthday = request.args.get('birthday')
    amount = request.args.get('amount')
    team= request.args.get('team')
    wdl = request.args.get('win_draw_loss')
    return render_template('confirm.html',first_name=first_name, last_name=last_name, birthday=birthday,
                           amount = amount,  team = team , win_draw_loss = wdl)


#############################################################################
# RUN APP
#############################################################################


if __name__ == "__main__":
    app.run(debug = True)