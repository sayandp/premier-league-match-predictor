import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("model_dataset.csv")

print("Dataset shape:", df.shape)

# -----------------------
# Check missing values
# -----------------------
print("\nMissing values:")
print(df.isnull().sum())

# Drop missing rows if any exist
df = df.dropna()

# -----------------------
# Encode categorical data
# -----------------------
le = LabelEncoder()

df["home_team"] = le.fit_transform(df["home_team"])
df["away_team"] = le.fit_transform(df["away_team"])

# -----------------------
# Separate features and target
# -----------------------
X = df.drop("result", axis=1)
y = df["result"]

# -----------------------
# Train / test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain set size:", X_train.shape)
print("Test set size:", X_test.shape)

# -----------------------
# Save processed datasets
# -----------------------
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPreprocessing completed")