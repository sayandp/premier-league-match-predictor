
import requests
import pandas as pd
import time

import os

API_KEY = os.getenv("API_FOOTBALL_KEY")

url = "https://v3.football.api-sports.io/fixtures"

headers = {
    "x-apisports-key": API_KEY
}

# Premier League league ID
league_id = 39

# seasons to download
seasons = list(range(2014, 2026))

all_matches = []

for season in seasons:

    print(f"Downloading season {season}")

    params = {
        "league": league_id,
        "season": season
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    matches = data["response"]

    for match in matches:

        if match["goals"]["home"] is None:
            continue

        all_matches.append({
            "season": season,
            "date": match["fixture"]["date"],
            "home_team": match["teams"]["home"]["name"],
            "away_team": match["teams"]["away"]["name"],
            "home_goals": match["goals"]["home"],
            "away_goals": match["goals"]["away"]
        })

    # avoid API rate limit
    time.sleep(1)

df = pd.DataFrame(all_matches)

df.to_csv("matches_2014_2025.csv", index=False)

print("Dataset saved as matches_2014_2025.csv")
print("Total matches:", len(df))
print(df.head())