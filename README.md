# ⚽ Premier League Match Predictor

A machine learning project that predicts Premier League match outcomes using historical match data, statistical modeling, and simulation techniques.

The project combines **machine learning, probabilistic modeling, and football analytics** to estimate match results and score probabilities.

---

# 📊 Project Overview

This project builds a football analytics system that predicts the outcome of a Premier League match between two teams.

The system uses historical match data to calculate team performance metrics and applies machine learning and statistical models to estimate match outcomes.

The interactive dashboard allows users to:

• Select two teams
• Predict match results
• View win probabilities
• Simulate possible scorelines
• Analyze team performance

---

# ⚙️ Features

### Machine Learning Prediction

A trained model predicts:

* Home Win
* Draw
* Away Win

based on engineered team performance features.

---

### Monte Carlo Match Simulation

Simulates thousands of matches to estimate the most likely scorelines.

Example output:

2–1 → 18%
1–0 → 14%
2–0 → 12%

---

### Poisson Goal Model

Football goals follow a distribution close to a **Poisson process**.

The model estimates the probability of each score combination.

Example heatmap:

Home Goals vs Away Goals probability matrix.

---

### Team Performance Analytics

The dashboard also shows:

• Expected goals comparison
• Last 5 match form
• Recent goal performance
• Head-to-head history

---

# 🧠 Machine Learning Model

Model used:

**XGBoost Classifier**

Why XGBoost:

• Handles nonlinear relationships
• Works well with tabular data
• Strong performance in sports prediction tasks

Target variable:

```
0 → Away Win
1 → Draw
2 → Home Win
```

---

# 📂 Project Structure

```
premier-league-match-predictor
│
├── app.py
├── data_collection.py
├── create_target.py
├── feature_engineering.py
├── preprocessing.py
├── train_model.py
│
├── football_model.pkl
├── matches_2014_2025.csv
│
├── assets/
│   └── fb.jpg
│
├── logos/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📥 Data Collection

Match data is collected using a football API.

The script:

```
data_collection.py
```

retrieves historical match results and stores them in:

```
matches_2014_2025.csv
```

Data includes:

• Home team
• Away team
• Goals scored
• Match results

---

# ⚙️ Feature Engineering

Features used for prediction include:

• Average goals scored
• Average goals conceded
• Team identifiers
• Historical performance metrics

Scripts used:

```
feature_engineering.py
preprocessing.py
```

---

# 🧪 Model Training

Training is done using:

```
train_model.py
```

The trained model is saved as:

```
football_model.pkl
```

This model is used by the Streamlit dashboard.

---

# 🖥️ Streamlit Dashboard

The interactive dashboard is built with the Python framework Streamlit.

Features:

• Team selection interface
• Match prediction
• Probability visualization
• Scoreline simulation
• Performance charts

---

# ▶️ Run the Project

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 2️⃣ Run the Streamlit app

```
streamlit run app.py
```

---

### 3️⃣ Open the dashboard

```
http://localhost:8501
```

---

# 📈 Example Dashboard Output

The dashboard displays:

• Match prediction
• Win probability bars
• Expected goals comparison
• Scoreline probability heatmap
• Monte Carlo simulation results

---

# 🎯 Future Improvements

Possible enhancements:

• ELO rating system for teams
• Player-level statistics
• Expected Goals (xG) model
• Live match predictions
• League table integration

---

# 👨‍💻 Author

**Sayand P**

Data Science / Sports Analytics Project

---

# ⭐ If you like the project

Consider giving the repository a star.
