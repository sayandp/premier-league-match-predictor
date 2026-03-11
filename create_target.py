import pandas as pd

df = pd.read_csv("matches_2014_2025.csv")

def get_result(row):
    
    if row["home_goals"] > row["away_goals"]:
        return 2
    
    elif row["home_goals"] < row["away_goals"]:
        return 0
    
    else:
        return 1


df["result"] = df.apply(get_result, axis=1)

df.to_csv("matches_with_results.csv", index=False)

print("Target variable created")
print(df.head())