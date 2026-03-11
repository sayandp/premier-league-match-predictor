import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import base64

# ------------------------------------------------
# Page configuration
# ------------------------------------------------
st.set_page_config(
    page_title="PL Football Match Predictor 2026",
    page_icon="⚽",
    layout="centered"
)

# ------------------------------------------------
# Load Background Image (FIX)
# ------------------------------------------------
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("assets/fb.jpg")

# ------------------------------------------------
# UI Styling
# ------------------------------------------------
st.markdown(f"""
<style>

.stApp {{
background-image: url("data:image/jpg;base64,{bg}");
background-size: cover;
background-position: center;
background-attachment: fixed;
}}

.block-container {{
background: rgba(0,0,0,0.75);
padding: 2rem;
border-radius: 15px;
}}

.title {{
text-align: center;
font-size: 42px;
font-weight: bold;
color: #00ffcc;
}}

.subtitle {{
text-align: center;
font-size: 18px;
margin-bottom: 25px;
color: white;
}}

.card {{
background: rgba(255,255,255,0.08);
backdrop-filter: blur(10px);
border-radius: 15px;
padding: 20px;
margin-top: 20px;
box-shadow: 0 0 20px rgba(0,255,204,0.3);
}}

.glow-bar {{
height: 18px;
border-radius: 10px;
background: linear-gradient(90deg,#00ffcc,#00ccff);
box-shadow: 0 0 10px #00ffcc;
margin-bottom: 10px;
}}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Title
# ------------------------------------------------
st.markdown('<div class="title">⚽ PL Football Match Predictor 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Football Analytics Dashboard</div>', unsafe_allow_html=True)

# ------------------------------------------------
# Load data
# ------------------------------------------------
model = joblib.load("football_model.pkl")
df = pd.read_csv("matches_2014_2025.csv")

teams = sorted(pd.concat([df["home_team"], df["away_team"]]).unique())
team_to_id = {team: i for i, team in enumerate(teams)}

# ------------------------------------------------
# Logo helper
# ------------------------------------------------
def team_logo(team):
    path = f"logos/{team}.png"
    if os.path.exists(path):
        return path
    return None

# ------------------------------------------------
# Team statistics
# ------------------------------------------------
def team_stats(team):

    matches = df[(df["home_team"] == team) | (df["away_team"] == team)]

    scored = []
    conceded = []

    for _, row in matches.iterrows():

        if row["home_team"] == team:
            scored.append(row["home_goals"])
            conceded.append(row["away_goals"])
        else:
            scored.append(row["away_goals"])
            conceded.append(row["home_goals"])

    return scored, conceded

# ------------------------------------------------
# Last 5 matches form
# ------------------------------------------------
def last5_form(team):

    matches = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(5)

    form = []
    goals = []

    for _, row in matches.iterrows():

        if row["home_team"] == team:

            goals.append(row["home_goals"])

            if row["home_goals"] > row["away_goals"]:
                form.append(3)
            elif row["home_goals"] == row["away_goals"]:
                form.append(1)
            else:
                form.append(0)

        else:

            goals.append(row["away_goals"])

            if row["away_goals"] > row["home_goals"]:
                form.append(3)
            elif row["away_goals"] == row["home_goals"]:
                form.append(1)
            else:
                form.append(0)

    return form, goals

# ------------------------------------------------
# Monte Carlo Simulation
# ------------------------------------------------
def simulate_match(home_avg, away_avg, simulations=10000):

    home_goals = np.random.poisson(home_avg, simulations)
    away_goals = np.random.poisson(away_avg, simulations)

    scorelines = {}

    for h, a in zip(home_goals, away_goals):

        score = f"{h}-{a}"

        if score not in scorelines:
            scorelines[score] = 0

        scorelines[score] += 1

    probs = {k: v / simulations for k, v in scorelines.items()}

    probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])

    return probs

# ------------------------------------------------
# Poisson Goal Model
# ------------------------------------------------
def poisson_model(home_xg, away_xg, max_goals=6):

    matrix = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):

            matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)

    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()

    return matrix, home_win, draw, away_win

# ------------------------------------------------
# Team selection
# ------------------------------------------------
home_team = st.selectbox("🏠 Home Team", teams)
away_team = st.selectbox("✈️ Away Team", teams)

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if st.button("⚽ Predict Match"):

    if home_team == away_team:
        st.error("Teams must be different")
        st.stop()

    home_scored, home_conceded = team_stats(home_team)
    away_scored, away_conceded = team_stats(away_team)

    home_avg_scored = np.mean(home_scored)
    away_avg_scored = np.mean(away_scored)

    home_avg_conceded = np.mean(home_conceded)
    away_avg_conceded = np.mean(away_conceded)

    match = pd.DataFrame([{
        "home_team": team_to_id[home_team],
        "away_team": team_to_id[away_team],
        "home_avg_scored": home_avg_scored,
        "home_avg_conceded": home_avg_conceded,
        "away_avg_scored": away_avg_scored,
        "away_avg_conceded": away_avg_conceded
    }])

    prediction = model.predict(match)[0]
    probs = model.predict_proba(match)[0]

    if prediction == 2:
        result = f"{home_team} likely to win"
    elif prediction == 0:
        result = f"{away_team} likely to win"
    else:
        result = "Match likely Draw"

    st.markdown("---")

    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        if team_logo(home_team):
            st.image(team_logo(home_team), width=90)

    with col2:
        st.subheader(f"{home_team} vs {away_team}")

    with col3:
        if team_logo(away_team):
            st.image(team_logo(away_team), width=90)

    st.success(result)

    # Win probability bars
    st.markdown("### Win Probability")

    st.markdown(f"""
    <div class="card">

    <b>{home_team} Win</b>
    <div class="glow-bar" style="width:{probs[2]*100}%"></div>
    {probs[2]*100:.1f}%

    <b>Draw</b>
    <div class="glow-bar" style="width:{probs[1]*100}%"></div>
    {probs[1]*100:.1f}%

    <b>{away_team} Win</b>
    <div class="glow-bar" style="width:{probs[0]*100}%"></div>
    {probs[0]*100:.1f}%

    </div>
    """, unsafe_allow_html=True)

    # Expected goals
    st.write("### ⚽ Expected Goals Comparison")

    fig, ax = plt.subplots()
    ax.bar([home_team, away_team], [home_avg_scored, away_avg_scored])
    st.pyplot(fig)

    # Last 5 form
    st.write("### 📊 Last 5 Match Form")

    home_form,_ = last5_form(home_team)
    away_form,_ = last5_form(away_team)

    st.bar_chart(pd.DataFrame({home_team: home_form, away_team: away_form}))

    # Monte Carlo
    st.write("### 🎯 Monte Carlo Score Prediction")

    scores = simulate_match(home_avg_scored, away_avg_scored)

    for score, prob in scores.items():
        st.write(f"{score} → {prob*100:.2f}%")

    # Poisson Model
    st.write("### 📊 Poisson Goal Model")

    matrix, home_win, draw_prob, away_win = poisson_model(home_avg_scored, away_avg_scored)

    st.write(f"{home_team} Win → {home_win*100:.2f}%")
    st.write(f"Draw → {draw_prob*100:.2f}%")
    st.write(f"{away_team} Win → {away_win*100:.2f}%")

    st.write("### 🔥 Score Probability Heatmap")

    fig, ax = plt.subplots()

    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu")

    ax.set_xlabel(f"{away_team} Goals")
    ax.set_ylabel(f"{home_team} Goals")

    st.pyplot(fig)