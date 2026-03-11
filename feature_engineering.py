import pandas as pd

df = pd.read_csv("matches_with_results.csv")

teams = pd.concat([df["home_team"], df["away_team"]]).unique()

team_stats = {}

for team in teams:

    matches = df[(df["home_team"] == team) | (df["away_team"] == team)]

    goals_scored = []
    goals_conceded = []

    for _, row in matches.iterrows():

        if row["home_team"] == team:
            goals_scored.append(row["home_goals"])
            goals_conceded.append(row["away_goals"])
        else:
            goals_scored.append(row["away_goals"])
            goals_conceded.append(row["home_goals"])

    team_stats[team] = {
        "avg_scored": sum(goals_scored) / len(goals_scored),
        "avg_conceded": sum(goals_conceded) / len(goals_conceded)
    }

home_avg_scored = []
home_avg_conceded = []
away_avg_scored = []
away_avg_conceded = []

for _, row in df.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    home_avg_scored.append(team_stats[home]["avg_scored"])
    home_avg_conceded.append(team_stats[home]["avg_conceded"])

    away_avg_scored.append(team_stats[away]["avg_scored"])
    away_avg_conceded.append(team_stats[away]["avg_conceded"])

df["home_avg_scored"] = home_avg_scored
df["home_avg_conceded"] = home_avg_conceded
df["away_avg_scored"] = away_avg_scored
df["away_avg_conceded"] = away_avg_conceded

df = df[[
    "home_team",
    "away_team",
    "home_avg_scored",
    "home_avg_conceded",
    "away_avg_scored",
    "away_avg_conceded",
    "result"
]]

df.to_csv("model_dataset.csv", index=False)

print("Feature engineering completed")
print(df.head())