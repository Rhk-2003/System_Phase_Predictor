# 🖥️ System Load Phase Predictor

> A multiclass machine learning pipeline that classifies a system's operational phase — **Normal**, **Moderate**, or **High** — using real-time resource utilization metrics like CPU usage, RAM consumption, thread count, and derived engineered features.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?style=flat-square&logo=streamlit)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.99-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 Live Demo

**[👉 Try the app on Streamlit Cloud](https://systemphasepredictor.streamlit.app/)**

Adjust CPU, RAM, thread count, and other system metrics interactively — the trained ML model predicts the system phase in real time with a radar chart visualisation.

---

## 📌 Overview

This project is **Activity 2** of a two-part system monitoring series. Building on the dataset collected and engineered in Activity 1, this activity transitions from *descriptive data collection* to *predictive machine learning* — training a classifier capable of forecasting system stress phases before they become critical.

The trained model is deployed as an interactive **Streamlit web application** that simulates live system metrics and displays the predicted operational phase in real time.

---

## 🎯 Objectives

- Preprocess a time-series dataset of ~8,400 records, handling missing values in rolling features and scaling continuous inputs
- Train and tune **at least two** traditional ML classifiers for comparative evaluation
- Achieve a **macro F1-score ≥ 0.85** across all three system phases for balanced prediction accuracy

---

## 📊 Dataset

**File:** `Thermal_experiment_dataset.csv`

| Feature | Description |
|---|---|
| `cpu_percent` | CPU utilization (%) |
| `ram_percent` | RAM utilization (%) |
| `process_count` | Number of active processes |
| `total_threads` | Total system thread count |
| `load_index` | Composite load metric (engineered) |
| `cpu_per_thread` | CPU usage normalized per thread (engineered) |
| `resource_pressure` | Combined resource stress score (engineered) |
| `rolling_cpu_mean` | Rolling average CPU (engineered, NaN at init rows) |
| `phase` | **Target** — `normal`, `moderate`, or `high` |

> **Source:** Collected during Activity 1 via a custom system monitoring script (~8,400 timestamped records).

---

## 🧠 Models Used

### Primary — Random Forest Classifier
Chosen for its robustness on tabular numerical data, resistance to overfitting via bagging, and built-in feature importance scores that reveal which metrics most influence phase transitions.

### Comparative — Gradient Boosting / XGBoost
Sequential tree boosting for benchmarking accuracy ceiling against Random Forest.

---

## ⚙️ Methodology

```
Thermal_experiment_dataset.csv
  └─► Drop timestamp / Encode target labels
        └─► Impute / Drop NaN (rolling_cpu_mean)
              └─► Train-Test Split (80/20, random_state=42)
                    └─► StandardScaler  →  system_scaler.pkl
                          └─► GridSearchCV (5-fold CV, macro F1)
                                └─► Best Model  →  system_phase_model.pkl
                                      └─► Streamlit app (app.py)
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

- **Macro F1 = 0.99** — far exceeds the 0.85 target with balanced accuracy across all three phases
- **`normal` phase** achieved perfect precision and recall (1.00), cleanly separating idle system states
- **`high` precision = 0.99** — when the model raises a HIGH load alert, it is correct 99% of the time, making it reliable for real-world alerting use cases
- No class suffered from class imbalance, validating the use of macro F1 over simple accuracy

---

## 🗂️ Repository Structure

```
├── Code.ipynb                      # Full ML pipeline: EDA, preprocessing, training & evaluation
├── Thermal_experiment_dataset.csv  # Dataset collected from Activity 1
├── app.py                          # Streamlit web application
├── system_phase_model.pkl          # Serialized trained Random Forest model
├── system_scaler.pkl               # Serialized StandardScaler
├── LICENSE
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Rhk-2003/system-load-phase-predictor.git
cd system-load-phase-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

### 4. Explore the notebook
Open `Code.ipynb` in Jupyter or VS Code to walk through the full ML pipeline.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
joblib
```

> Generate your own lockfile with: `pip freeze > requirements.txt`

---

## 🔗 Continuity from Activity 1

This project is a direct continuation of **Activity 1**, where the system monitoring dataset was collected and feature-engineered. The features used here — specifically `load_index`, `resource_pressure`, and `rolling_cpu_mean` — were derived during Activity 1's data extraction phase. Activity 2 transitions from descriptive data collection to predictive ML analysis using the exact same ~8,400-record dataset, now deployed as a live interactive application.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙋 Author

**Rhk-2003**  
[GitHub Profile](https://github.com/Rhk-2003)
