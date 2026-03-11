import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# -------------------------
# Load processed data
# -------------------------
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# -------------------------
# Model
# -------------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

# -------------------------
# Train
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# Predict
# -------------------------
preds = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
print("Accuracy:", accuracy_score(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

# -------------------------
# Save model
# -------------------------
joblib.dump(model, "football_model.pkl")

print("\nModel saved as football_model.pkl")
import matplotlib.pyplot as plt

importance = model.feature_importances_

plt.bar(X_train.columns, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()