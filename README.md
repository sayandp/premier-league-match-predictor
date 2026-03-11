# ⚽ Premier League Football Match Predictor (Machine Learning + Sports Analytics)

An end-to-end **data science project** that predicts Premier League football match outcomes using historical match data, statistical modeling, and machine learning.

The project also includes an **interactive analytics dashboard** built with Streamlit that visualizes predictions, probabilities, and team performance.

---

# 📊 Project Overview

Predicting football match outcomes is a classic problem in **sports analytics and predictive modeling**.

This project builds a complete pipeline that:

• Collects football match data using an API
• Cleans and preprocesses the data
• Engineers predictive features
• Trains a machine learning model
• Uses statistical simulations for score prediction
• Displays results in an interactive Streamlit dashboard

The system predicts:

* Match outcome (Home Win / Draw / Away Win)
* Win probabilities
* Expected goals comparison
* Most likely scorelines
* Score probability heatmaps

---

# 🔗 Data Collection (API)

Historical match data is collected using a football statistics API.

The script `data_collection.py` connects to the API and downloads match data including:

* Home team
* Away team
* Goals scored
* Goals conceded
* Match date
* League information

Example API request:

```python
headers = {
    "x-apisports-key": API_KEY
}
```

The collected data is saved as:

```
matches_2014_2025.csv
```

This automated pipeline allows the dataset to be updated easily for new seasons.

---

# 🧹 Data Preprocessing

Raw sports data requires cleaning and transformation before it can be used for modeling.

The preprocessing stage performs:

### Data Cleaning

* Removing incomplete matches
* Handling missing values
* Standardizing team names

### Feature Engineering

New features are created to represent team strength and performance:

```
home_avg_scored
home_avg_conceded
away_avg_scored
away_avg_conceded
```

These features capture:

* Team attacking strength
* Defensive ability
* Goal scoring patterns

### Dataset Preparation

The processed dataset is split into training and testing sets:

```
X_train.csv
X_test.csv
y_train.csv
y_test.csv
```

---

# 🧠 Machine Learning Model

A classification model is trained to predict match outcomes.

Predicted classes:

```
Home Win
Draw
Away Win
```

The model learns patterns in team performance and scoring behavior.

After training, the model is saved as:

```
football_model.pkl
```

This file is used directly by the dashboard for predictions.

---

# 🎯 Monte Carlo Match Simulation

Monte Carlo simulation generates thousands of possible match outcomes using probabilistic goal scoring.

Example simulated results:

```
2-1 → 18.4%
1-1 → 16.2%
1-0 → 12.7%
2-0 → 11.1%
0-1 → 9.3%
```

This provides realistic **scoreline probability predictions**.

---

# 📊 Poisson Goal Model

Football goals are commonly modeled using a **Poisson distribution**.

The Poisson model estimates the probability of scoring **0–5 goals per team**.

From the probability matrix we compute:

* Home win probability
* Draw probability
* Away win probability

Example output:

```
Home Win → 52.1%
Draw → 26.3%
Away Win → 21.6%
```

---

# 🔥 Score Probability Heatmap

The Poisson model generates a **score probability heatmap** showing the likelihood of each scoreline.

Example matrix:

| Home\Away | 0    | 1    | 2    | 3    |
| --------- | ---- | ---- | ---- | ---- |
| 0         | 0.07 | 0.09 | 0.05 | 0.02 |
| 1         | 0.11 | 0.14 | 0.08 | 0.03 |
| 2         | 0.08 | 0.10 | 0.06 | 0.02 |

This visualization is widely used in **football analytics and betting models**.

---

# 📈 Interactive Dashboard

The project includes a dashboard built using
Streamlit.

The dashboard allows users to:

* Select two teams
* Generate match predictions
* Visualize probabilities
* Explore team statistics

### Dashboard Features

✔ Match outcome prediction
✔ Win probability visualization
✔ Expected goals comparison
✔ Monte Carlo score simulation
✔ Poisson score probability heatmap
✔ Team form analysis (last 5 matches)
✔ Head-to-head match history
✔ Football themed UI with team logos

---

# 🗂 Project Structure

```
FTBALL/
│
├── app.py
│   Streamlit dashboard application
│
├── data_collection.py
│   Collects match data from football API
│
├── feature_engineering.py
│   Creates team performance features
│
├── preprocessing.py
│   Data cleaning and dataset preparation
│
├── train_model.py
│   Machine learning model training
│
├── football_model.pkl
│   Trained prediction model
│
├── matches_2014_2025.csv
│   Historical match dataset
│
├── requirements.txt
│   Python dependencies
│
├── assets/
│   Dashboard images and UI assets
│
└── logos/
    Premier League team logos
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/sayandp/football-match-predictor.git
```

Navigate to the project folder:

```
cd football-match-predictor
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

Start the Streamlit application:

```
streamlit run app.py
```

Open in your browser:

```
http://localhost:8501
```

---

# 🚀 Future Improvements

Possible improvements for the project:

* Elo rating based team strength model
* Player level statistics integration
* Expected Goals (xG) dataset
* Live match predictions using API
* Hyperparameter tuning for ML model
* Model evaluation across seasons

---

# 🎯 Skills Demonstrated

This project demonstrates several data science skills:

* API data collection
* Data cleaning and preprocessing
* Feature engineering
* Machine learning modeling
* Statistical modeling
* Monte Carlo simulation
* Data visualization
* Interactive dashboard development

---

# 👨‍💻 Author

**Sayand P**

Data Science / Sports Analytics Project
