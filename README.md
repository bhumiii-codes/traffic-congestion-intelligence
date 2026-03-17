#  Urban Traffic Congestion Intelligence System

> A machine learning-powered traffic intelligence platform that predicts congestion levels, identifies bottlenecks, and provides real-time insights through an interactive web dashboard.

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

##  Problem Statement

Urban traffic congestion causes significant economic losses, increased pollution, and longer commute times. Traditional traffic systems only count vehicles — they don't predict or explain congestion.

This system solves that by:
- **Predicting** exact traffic volume (regression)
- **Classifying** congestion severity (Low / Moderate / High / Critical)
- **Explaining** which factors contribute most to congestion
- **Visualizing** patterns through a live interactive dashboard

---

##  Objectives

- Predict traffic congestion levels at different times of the day
- Identify high-risk congestion periods and contributing factors
- Provide data-driven insights for urban traffic management
- Analyze the impact of weather, holidays, and time patterns on congestion

---

##  Dataset

**Source:** [Metro Interstate Traffic Volume — Kaggle](https://www.kaggle.com/datasets/robikscube/metro-interstate-traffic-volume)

| Property | Value |
|---|---|
| Location | Metro Interstate I-94, Minneapolis–St. Paul, USA |
| Time Period | October 2012 — September 2018 |
| Records | 48,204 hourly observations |
| Features | 9 original → 37 engineered |
| Target (Regression) | `traffic_volume` (vehicles/hr) |
| Target (Classification) | `congestion_level` (Low/Moderate/High/Critical) |

---

##  Project Structure
```
traffic-congestion-intelligence/
│
├── data/
│   ├── raw/                          # Original dataset (CSV)
│   └── processed/                    # Cleaned & processed data
│
├── src/
│   ├── data/
│   │   ├── loader.py                 # Load & validate raw data
│   │   └── preprocessor.py          # Clean, encode, handle missing values
│   ├── features/
│   │   └── engineer.py              # 37 feature engineering functions
│   ├── models/
│   │   ├── trainer.py               # Train & cross-validate all models
│   │   └── evaluator.py             # Metrics, plots, robustness checks
│   └── visualization/
│       ├── plots.py                 # Static matplotlib visualizations
│       └── dashboard.py             # Flask web dashboard & APIs
│
├── templates/
│   └── index.html                   # Interactive multi-page dashboard UI
│
├── outputs/
│   ├── models/                      # Saved trained models (.pkl)
│   └── figures/                     # All generated charts & visualizations
│
├── main.py                          # Full pipeline runner
├── save_cls_models.py               # Save classification models & metrics
├── requirements.txt                 # All dependencies
└── README.md
```

---

##  ML Pipeline
```
Raw CSV Data
     ↓
1. Data Loading & Validation
     ↓
2. Preprocessing
   • Handle missing values (median/mode imputation)
   • Remove duplicates (17 rows)
   • Outlier removal (IQR method)
   • Weather & holiday encoding
   • Congestion label generation
     ↓
3. Feature Engineering (37 features)
   • Cyclical time encoding (sin/cos for hour, month, day of week)
   • Data-driven rush hour detection
   • Weather severity flags
   • Lag features (1hr, 2hr, 24hr, 168hr)
   • Rolling averages (3hr, 24hr)
     ↓
4. Train / Validation / Test Split (70% / 15% / 15%)
     ↓
5. Model Training (4 algorithms × 2 tasks)
     ↓
6. Cross Validation (5-Fold)
     ↓
7. Evaluation (MAE, RMSE, R², F1, Precision, Recall, Confusion Matrix)
     ↓
8. Visualization & Dashboard
```

---

##  Algorithms

### Regression (Predict Exact Volume)

| Model | Purpose |
|---|---|
| Linear Regression | Baseline — simple linear relationship |
| Random Forest | Ensemble of 100 trees — best performer |
| Gradient Boosting | Sequential boosting — corrects previous errors |
| XGBoost | Optimized boosting with L1/L2 regularization |

### Classification (Predict Congestion Level)

Same 4 algorithms applied as classifiers to predict:
`Low` / `Moderate` / `High` / `Critical`

---

##  Results

### Regression Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~650 | ~820 | ~0.82 |
| Gradient Boosting | 180.9 | ~220 | 0.9801 |
| XGBoost | 163.6 | ~200 | 0.9839 |
| **Random Forest** | **124.7** | **~160** | **0.988** |

### Classification Performance

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.75 | ~0.74 | ~0.75 | ~0.75 |
| Gradient Boosting | High | High | High | High |
| XGBoost | High | High | High | High |
| **Random Forest** | **Best** | **Best** | **Best** | **Best** |

> Full metrics available in `outputs/model_metrics.json` after running the pipeline.

---

##  Feature Engineering Highlights

### Cyclical Time Encoding
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```
Preserves continuity between hour 23 and hour 0.

### Data-Driven Rush Hour Detection
```python
# Learn peak hours FROM the data — not hardcoded assumptions
am_peaks = weekday_hourly[weekday_hourly.index < 12].nlargest(3).index
pm_peaks = weekday_hourly[weekday_hourly.index >= 12].nlargest(3).index
```

### Lag Features
```python
volume_lag_1h   # Traffic 1 hour ago
volume_lag_24h  # Same hour yesterday
volume_lag_168h # Same hour last week
rolling_mean_3h # Average of last 3 hours
```

---

##  Dashboard Features

| Page | Features |
|---|---|
|  Overview | Stat cards, hourly line chart, congestion donut |
|  Heatmap | Hour×Day heatmap, risk score table |
|  Weather & Trends | Weather impact, temperature curve, year-over-year |
|  Model Comparison | R², MAE, F1, accuracy charts for all models |
|  Live Predictions | Real-time ML prediction, risk calculator, animated timeline |
|  Smart Alerts | Top 5 high-risk windows, congestion risk bar chart |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/bhumiii-codes/traffic-congestion-intelligence.git
cd traffic-congestion-intelligence
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/robikscube/metro-interstate-traffic-volume) and place at:
```
data/raw/Metro_Interstate_Traffic_Volume.csv
```

### 5. Run the full ML pipeline
```bash
python main.py
```

### 8. Launch the dashboard
```bash
python -m src.visualization.dashboard
```

### 9. Open in browser
```
http://127.0.0.1:5000
```

---

##  Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
joblib
jupyter
flask
```

---

##  Key Insights

- **Temporal features dominate** — hour of day explains ~66% of model importance
- **PM rush (17:00) is worst** — 81.6% of weekday records are High/Critical
- **Snow reduces traffic by 32%** compared to clear conditions
- **Holidays cut volume ~45%** below weekly average
- **Lag features boosted R²** from ~0.85 to 0.988
- **Weekends 28% lower** than weekdays on average

---

##  Team

Built by Bhumika Singh, Kaveri Patle and Megan Sheel


