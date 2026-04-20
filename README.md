# 🖥️ System Load Phase Predictor

> A multiclass machine learning pipeline that classifies a system's operational phase — **Normal**, **Moderate**, or **High** — using real-time resource utilization metrics like CPU usage, RAM consumption, thread count, and derived engineered features.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.99-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Overview

This project is **Activity 2** of a two-part system monitoring series. Building on the dataset collected and engineered in Activity 1, this activity transitions from *descriptive data collection* to *predictive machine learning* — training a classifier capable of forecasting system stress phases before they become critical.

The model is paired with an **interactive System Load Dashboard** (shown below) that simulates live metrics and displays the predicted phase in real time.

![System Load Dashboard](./assets/dashboard_preview.png)

---

## 🎯 Objectives

- Preprocess a time-series dataset of ~8,400 records, handling missing values in rolling features and scaling continuous inputs
- Train and tune **at least two** traditional ML classifiers for comparative evaluation
- Achieve a **macro F1-score ≥ 0.85** across all three system phases for balanced prediction accuracy

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `cpu_percent` | CPU utilization (%) |
| `ram_percent` | RAM utilization (%) |
| `process_count` | Number of active processes |
| `total_threads` | Total system thread count |
| `load_index` | Composite load metric (engineered) |
| `cpu_per_thread` | CPU usage normalized per thread (engineered) |
| `resource_pressure` | Combined resource stress score (engineered) |
| `rolling_cpu_mean` | Rolling average CPU (engineered, contains NaN at init) |
| `phase` | **Target** — `normal`, `moderate`, or `high` |

> **Source:** Collected during Activity 1 via a custom system monitoring script. ~8,400 timestamped records.

---

## 🧠 Models Used

### Primary — Random Forest Classifier
Chosen for its robustness on tabular numerical data, resistance to overfitting via bagging, and built-in feature importance scores that reveal which metrics most influence phase transitions.

### Comparative — Gradient Boosting / XGBoost
Sequential tree boosting to correct residual errors from prior trees — provides a strong accuracy ceiling for benchmarking against Random Forest.

---

## ⚙️ Methodology

```
Raw CSV
  └─► Drop timestamp / Encode target labels
        └─► Impute / Drop NaN (rolling_cpu_mean)
              └─► Train-Test Split (80/20)
                    └─► StandardScaler
                          └─► GridSearchCV (5-fold CV, macro F1)
                                └─► Best Model → Evaluation
```

### Hyperparameter Grid (Random Forest)

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
```

---

## 📈 Results

### Classification Report

```
              precision    recall  f1-score   support

        high       0.99      0.99      0.99       551
    moderate       0.99      0.99      0.99       566
      normal       1.00      1.00      1.00       563

    accuracy                           0.99      1680
   macro avg       0.99      0.99      0.99      1680
weighted avg       0.99      0.99      0.99      1680
```

### Key Takeaways

- **Macro F1 = 0.99** — far exceeding the 0.85 target, with balanced accuracy across all three phases
- **`normal` phase** achieved perfect precision and recall (1.00), indicating the model cleanly separates idle system states
- **`high` precision = 0.99** — when the model raises a HIGH load alert, it is correct 99% of the time (very low false positive rate), making it reliable for real-world alerting
- No class suffered significantly from the class imbalance, validating the use of macro F1 over simple accuracy

---

## 🗂️ Project Structure

```
system-load-phase-predictor/
│
├── data/
│   └── system_metrics.csv          # Raw dataset from Activity 1
│
├── notebooks/
│   └── activity2_ml_pipeline.ipynb # Full EDA + training notebook
│
├── src/
│   ├── preprocess.py               # Data cleaning & feature scaling
│   ├── train.py                    # Model training & GridSearchCV
│   └── evaluate.py                 # Metrics, confusion matrix, report
│
├── dashboard/
│   └── index.html                  # Interactive System Load Dashboard
│
├── assets/
│   └── dashboard_preview.png       # Dashboard screenshot
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/system-load-phase-predictor.git
cd system-load-phase-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the training pipeline
```bash
python src/train.py
```

### 4. View results
```bash
python src/evaluate.py
```

### 5. Open the dashboard
Open `dashboard/index.html` in any browser to interact with the live simulator.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyter
```

---

## 🔗 Continuity from Activity 1

This project is a direct continuation of **Activity 1**, where the system monitoring dataset was collected and feature-engineered. The features used here — specifically `load_index`, `resource_pressure`, and `rolling_cpu_mean` — were derived during Activity 1's data extraction phase. Activity 2 successfully transitions from descriptive data collection to predictive ML analysis using the exact same ~8,400-record dataset.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙋 Author

**Your Name**  
Feel free to open an issue or pull request for improvements!
