# 📊 End-to-End ML Pipeline — Telco Customer Churn Prediction

> **Internship Task 2** · Scikit-learn Pipeline API · GridSearchCV · joblib

---

## 🎯 Objective

Build a **reusable, production-ready machine learning pipeline** to predict whether a telecom customer will churn. The pipeline encapsulates preprocessing, model training, and evaluation using Scikit-learn's `Pipeline` and `ColumnTransformer` APIs.

---

## 📁 Repository Structure

```
telco-churn-ml-pipeline/
├── telco_churn_pipeline.ipynb       # Jupyter Notebook (with outputs)
├── Telco-Customer-Churn.csv         # Dataset
├── lr_churn_pipeline.joblib         # Exported Logistic Regression pipeline
├── rf_churn_pipeline.joblib         # Exported Random Forest pipeline
└── README.md
```

---

## 📦 Dataset

**IBM Telco Customer Churn Dataset**
- 7,043 customer records × 21 features
- Target: `Churn` (Yes / No) — ~26% positive class
- Features: demographics, account details, service subscriptions

Download from: [IBM Telco Churn on GitHub](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

---

## 🛠️ Methodology / Approach

### 1. Data Preprocessing
| Step | Detail |
|------|--------|
| Fix `TotalCharges` | Coerce whitespace to NaN → fill with median |
| Drop `customerID` | Non-predictive identifier |
| Encode target | `Churn`: Yes → 1, No → 0 |
| Numerical features | `SimpleImputer(median)` → `StandardScaler` |
| Categorical features | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

### 2. Pipeline Architecture

```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([imputer, scaler]),    numerical_cols),
        ("cat", Pipeline([imputer, encoder]),   categorical_cols)
    ])),
    ("classifier", LogisticRegression() / RandomForestClassifier())
])
```

### 3. Models Trained
- **Logistic Regression** — baseline, interpretable, fast
- **Random Forest** — ensemble, captures non-linear patterns

### 4. Hyperparameter Tuning
- **Method:** `GridSearchCV` with 5-fold `StratifiedKFold`
- **Scoring metric:** `roc_auc` (appropriate for imbalanced classes)

| Model | Parameters Tuned |
|-------|-----------------|
| Logistic Regression | `C` ∈ {0.01, 0.1, 1, 10}, `solver` ∈ {liblinear, lbfgs} |
| Random Forest | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20}, `min_samples_split` ∈ {2, 5} |

### 5. Export
Both tuned pipelines are exported using `joblib.dump()` and can be reloaded for inference without any manual preprocessing:

```python
import joblib
model = joblib.load("rf_churn_pipeline.joblib")
predictions = model.predict(raw_dataframe)  # handles preprocessing automatically
```

---

## 📈 Key Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.8048 | 0.6032 | **0.8411** |
| Random Forest       | 0.7999 | 0.5778 | 0.8385 |

Best hyperparameters found:
- **LR:** `C=10`, `solver=liblinear`
- **RF:** `n_estimators=200`, `max_depth=10`, `min_samples_split=5`

---

## 🔍 Key Observations

1. **Logistic Regression** achieves a slightly higher ROC-AUC (0.841), making it the best model for this task.
2. **Top churn predictors:** `Contract` type (month-to-month customers churn most), low `tenure`, and high `MonthlyCharges`.
3. **Pipeline design prevents data leakage** — the scaler and encoder are fit only on training data and applied consistently to the test set.
4. **Class imbalance** (~26% churn) is handled by using stratified splits and ROC-AUC as the primary metric.
5. The **joblib-exported pipelines** are production-ready: load and serve them directly in any Python environment.

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn joblib matplotlib seaborn

# Run the script
python telco_churn_pipeline.py

# Or open the notebook
jupyter notebook telco_churn_pipeline.ipynb
```

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow)

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Pipeline, GridSearchCV, models, metrics |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualizations |
| `joblib` | Model serialization |

---

## 📚 Skills Demonstrated

- ✅ ML pipeline construction with `Pipeline` + `ColumnTransformer`
- ✅ Hyperparameter tuning with `GridSearchCV`
- ✅ Model evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
- ✅ Model export and reusability with `joblib`
- ✅ Production-readiness practices (no leakage, modular code)

---

*Internship Task 2 — Machine Learning Engineering*
