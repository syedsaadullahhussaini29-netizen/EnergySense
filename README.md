# EnergySense AI

## Smart Energy Consumption Prediction & Sustainability Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org)
[![SDG 7](https://img.shields.io/badge/SDG-7%20Clean%20Energy-yellow)](https://sdgs.un.org/goals/goal7)
[![SDG 13](https://img.shields.io/badge/SDG-13%20Climate%20Action-green)](https://sdgs.un.org/goals/goal13)

---

## Project Overview

**EnergySense AI** is an end-to-end machine learning application that predicts household appliance energy consumption and helps users understand energy consumption patterns. It transforms the UCI Appliances Energy Prediction dataset into a professional, interactive Streamlit dashboard.

---

## Problem Statement

> *How can machine learning be used to predict household appliance energy consumption from indoor environmental conditions, weather conditions, lighting, and time-related factors, while providing useful insights that encourage energy efficiency and responsible consumption?*

The project focuses on:
1. **Predicting** appliance energy consumption from environmental and temporal features.
2. **Understanding** consumption patterns through interactive visualisations and scenario simulation.

---

## Objectives

- Predict appliance energy consumption (Wh per 10-minute interval) using regression models.
- Compare multiple machine learning models with a chronological (time-aware) evaluation.
- Provide interactive data exploration, pattern analysis, and what-if scenario simulation.
- Present sustainability insights aligned with SDG 7, 11, 12, and 13.

---

## Dataset

**UCI Appliances Energy Prediction**

| Property | Value |
|---|---|
| Records | 19,735 observations |
| Frequency | Every 10 minutes |
| Period | January 11 – May 27, 2016 |
| Location | A house in Stambruges, Belgium |
| Target column | `Appliances` (Wh per 10-min interval) |
| Missing values | None |
| Duplicate rows | None |
| Columns | 29 total (date, Appliances, lights, 9 indoor temperature/humidity pairs, outdoor weather variables, 2 random noise columns) |

**Official source:**
https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

**Citation:**
Candanedo, L. (2017). *Appliances Energy Prediction*. UCI Machine Learning Repository. DOI: [10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G)

---

## Features

| Column | Description |
|---|---|
| `date` | Timestamp (10-minute intervals) |
| `Appliances` | Energy consumed by appliances (Wh) — **target** |
| `lights` | Energy consumed by lighting (Wh) |
| `T1`–`T9` | Indoor temperature readings from 9 rooms/locations (°C) |
| `RH_1`–`RH_9` | Indoor relative humidity for 9 rooms (%) |
| `T_out` | Outdoor temperature (°C) |
| `Press_mm_hg` | Atmospheric pressure (mm Hg) |
| `RH_out` | Outdoor relative humidity (%) |
| `Windspeed` | Wind speed (m/s) |
| `Visibility` | Visibility (km) |
| `Tdewpoint` | Dew point temperature (°C) |
| `rv1`, `rv2` | Random noise variables (excluded from modelling) |

---

## Feature Engineering

Features engineered from the date/time column:

| Feature | Description |
|---|---|
| `hour`, `minute` | Hour and minute of the observation |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `month`, `day` | Month and day of month |
| `is_weekend` | 1 if Saturday or Sunday |
| `part_of_day` | Morning / Late Morning / Afternoon / Evening / Night |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour |
| `dow_sin`, `dow_cos` | Cyclical encoding of day of week |
| `month_sin`, `month_cos` | Cyclical encoding of month |

Lag features (past appliance consumption — no target leakage):

| Feature | Description |
|---|---|
| `appliances_lag_N` | Target value N × 10 minutes ago |
| `appliances_roll_mean_W` | Rolling mean over W × 10-minute windows |
| `appliances_roll_std_W` | Rolling std over W × 10-minute windows |

Lag values used: 1, 2, 3, 6, 12, 18, 24, 36, 48, 72, 144 steps.
Rolling windows: 6, 12, 24, 36, 144 steps.

All lag/rolling features use `.shift(1)` or higher — no future information is used.

---

## Machine Learning Methodology

### Models Compared

| Model | Key Settings |
|---|---|
| Linear Regression | StandardScaler pipeline |
| Random Forest | depth=6, min_samples_leaf=30, n_estimators=200 |
| Gradient Boosting | learning_rate=0.02, depth=3, min_samples_leaf=30, n_estimators=300 |

XGBoost is included automatically if installed.

### Train/Test Strategy

A **chronological split** (no shuffling) is used:
- **Training set:** Earliest 80% of observations (Jan–Apr 2016)
- **Test set:** Latest 20% of observations (May 2016)

This avoids temporal data leakage and produces a more realistic evaluation.

### Best Model Selection

The model with the **lowest test-set RMSE** is automatically selected and saved.

---

## Results

| Model | MAE (Wh) | RMSE (Wh) | Test R² |
|---|---|---|---|
| Random Forest | 26.07 | 57.69 | 0.569 |
| Gradient Boosting | 28.75 | 57.93 | 0.565 |
| Linear Regression | 27.70 | 58.67 | 0.554 |

*All models beat the naive mean baseline. Lag-1 (10-min past consumption) is the most important feature (importance ≈ 0.60 for tree models).*

---

## Dashboard Features

| Page | Description |
|---|---|
| 🏠 Home | KPI cards, problem statement, dataset overview, quick chart |
| 📊 Data Explorer | Sample data, statistics, column info, distributions |
| 📈 Consumption Analysis | Time series, hourly/weekly patterns, environmental relationships |
| 🤖 ML Prediction | Interactive form to get a model-estimated consumption prediction |
| 🔍 Model Performance | Model comparison, actual vs predicted, residuals, feature importance |
| 💡 What-If Simulator | Compare baseline vs modified scenario |
| ⚠️ Energy Insights | High-consumption observations analysis |
| 🌍 Sustainability & SDGs | SDG 7, 11, 12, 13 alignment |
| ℹ️ About | Full project documentation |

---

## SDG Alignment

| SDG | Connection |
|---|---|
| ⚡ SDG 7 — Affordable and Clean Energy | Supports awareness of household electricity consumption and energy efficiency |
| 🏙️ SDG 11 — Sustainable Cities | Provides insights relevant to smarter energy monitoring in buildings |
| ♻️ SDG 12 — Responsible Consumption | Understanding consumption patterns can support more conscious energy use |
| 🌱 SDG 13 — Climate Action | Energy efficiency awareness as one contributing factor |

*Disclaimer: This project supports awareness and provides insights. It does not directly achieve any SDG.*

---

## Project Architecture

```
EnergySenseAI/
│
├── app.py                   ← Streamlit dashboard
├── requirements.txt
├── README.md
│
├── data/
│   └── energydata_complete.csv   ← UCI dataset
│
├── models/
│   ├── best_model.joblib         ← Trained best model
│   ├── model_metrics.json        ← Evaluation metrics for all models
│   └── feature_columns.json      ← Feature column list
│
├── src/
│   ├── preprocessing.py          ← Data loading, validation, cleaning
│   ├── feature_engineering.py    ← Temporal and lag feature creation
│   ├── train.py                  ← End-to-end training pipeline
│   ├── evaluate.py               ← MAE / RMSE / R² utilities
│   └── explain.py                ← Feature importance utilities
│
└── notebooks/
    └── energy_analysis.ipynb     ← Exploratory analysis notebook
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EnergySenseAI.git
cd EnergySenseAI

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

Ensure the dataset is placed at `data/energydata_complete.csv`.

---

## How to Run Locally

### Step 1: Train the models

```bash
python src/train.py
```

This will:
- Load and clean the dataset
- Engineer features (including lag features)
- Split data chronologically (80/20)
- Train Linear Regression, Random Forest, and Gradient Boosting
- Select the best model by test RMSE
- Save artifacts to `models/`

### Step 2: Launch the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Streamlit Community Cloud Deployment

1. Push the project to a public GitHub repository (include `models/` directory with saved artifacts, or retrain on first run).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repository.
3. Set the main file to `app.py`.
4. Add `requirements.txt` — Streamlit Cloud installs dependencies automatically.
5. Ensure `data/energydata_complete.csv` is in the repository (the app loads it at startup).
6. Deploy.

**Note:** All paths in the application use relative paths — no absolute paths are used. The app is compatible with Streamlit Community Cloud.

---

## Limitations

- The dataset covers a single house over approximately 4.5 months. Results may not generalise to other households, climates, or building types.
- The model predicts based on historical patterns. It is an estimate, not a real measurement.
- The chronological test set (May 2016) shows some seasonal differences from training months (Jan–Apr), which limits generalisation.
- Lag features require recent past consumption values — unavailable in cold-start scenarios.
- Correlation between features and target does not imply causation.
- External factors (occupant behaviour, appliance upgrades) are not represented in the dataset.

---

## Future Improvements

- Include data from multiple households for better generalisation.
- Explore LSTM/sequence models for time-series prediction.
- Add real-time prediction API connected to live sensor data.
- Incorporate energy tariff data to provide cost-based insights.
- Build occupancy-aware models.
- Extend the SDG section with energy benchmark comparisons.

---

## Dataset Citation

Candanedo, L. (2017). *Appliances Energy Prediction*. UCI Machine Learning Repository.
DOI: [10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G)

Dataset source: https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

---

## License

This project is for educational and research purposes. The UCI dataset is provided under its
original terms — see the [UCI ML Repository](https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction) for licensing information.
