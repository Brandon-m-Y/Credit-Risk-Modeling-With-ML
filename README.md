# 💰 Credit Risk Modeling Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> End-to-end credit risk modeling pipeline using Python to predict loan default probability on Kaggle's German Credit Risk dataset — from raw data to a live Streamlit prediction app.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [End-to-End Analysis & Modeling Process](#2-end-to-end-analysis--modeling-process)
   - [2.1 Business Understanding & Problem Definition](#21-business-understanding--problem-definition)
   - [2.2 Data Acquisition & Initial Inspection](#22-data-acquisition--initial-inspection)
   - [2.3 Exploratory Data Analysis](#23-exploratory-data-analysis)
   - [2.4 Data Preprocessing & Feature Engineering](#24-data-preprocessing--feature-engineering)
   - [2.5 Multi-Model Training & Hyperparameter Tuning](#25-multi-model-training--hyperparameter-tuning)
   - [2.6 Model Evaluation & Comparison](#26-model-evaluation--comparison)
   - [2.7 Deployment — Streamlit Prediction App](#27-deployment--streamlit-prediction-app)
3. [Repository Structure](#3-repository-structure)
4. [Quick Start](#4-quick-start)
5. [Reproducibility & Extensions](#5-reproducibility--extensions)
6. [License & Acknowledgements](#6-license--acknowledgements)

---

## 1. Project Overview

Credit default is one of the most costly risks a lending institution faces. When a borrower fails to repay, lenders absorb not only the outstanding principal but also the operational costs of collections, regulatory capital charges, and reputational damage. Accurately identifying high-risk applicants before a loan is issued is therefore a core function of any modern credit underwriting operation.

This project builds a production-oriented credit risk scoring pipeline on the **German Credit Risk dataset** (available on Kaggle). It replicates the full analytical workflow a data science team at a bank or fintech would follow: ingesting raw applicant data, performing rigorous exploratory analysis, engineering features, benchmarking multiple classification algorithms with cross-validated hyperparameter search, and surfacing the winning model through an interactive web interface.

The primary objective is to classify each applicant as **Good** (low default risk) or **Bad** (high default risk) using demographic and financial attributes, while explicitly accounting for class imbalance — a structural reality in any credit portfolio where defaults are less frequent than repayments.

**Target audience:** Lending institutions, fintech credit teams, community banks, and analytics consultants evaluating ML-driven underwriting tools.

---

## 2. End-to-End Analysis & Modeling Process

### 2.1 Business Understanding & Problem Definition

| Dimension | Detail |
|---|---|
| **Business Problem** | Predict loan default probability to reduce credit losses and improve underwriting decisions |
| **ML Task** | Binary classification (Good Risk = 1 / Bad Risk = 0) |
| **Dataset** | German Credit Risk — 1,000 applicants, 10 features + 1 target |
| **Success Metric** | Accuracy on held-out test set; class-balanced model selection |
| **Deployment Target** | Real-time scoring via Streamlit web application |

Credit decisions are asymmetric: approving a bad borrower (false negative) typically costs far more than rejecting a good one (false positive). The pipeline therefore uses **class-balanced training** (`class_weight='balanced'`, `scale_pos_weight`) throughout to prevent models from naively predicting the majority class.

---

### 2.2 Data Acquisition & Initial Inspection

The raw dataset contains **1,000 rows** across 10 features and a binary target (`Risk: good/bad`). Initial inspection revealed:

- **No duplicate rows** in the dataset.
- **Two features with missing values**: `Saving accounts` (183 nulls, 18.3%) and `Checking account` (394 nulls, 39.4%).
- **Significant class imbalance** in the full dataset: 70% Good / 30% Bad.
- After dropping rows with any missing account information, the working dataset is **522 records** with a near-balanced split of 55.7% Good / 44.3% Bad.

```
Original dataset:  1,000 rows × 11 columns
After null removal:  522 rows × 10 columns
Train set:  417 samples (80%)
Test set:    105 samples (20%) — stratified split
```

**Key feature summary:**

| Feature | Type | Notes |
|---|---|---|
| `Age` | Numeric | 19–75, mean 35.5 |
| `Sex` | Categorical | 69% male |
| `Job` | Ordinal (0–3) | Skill level; higher = more skilled |
| `Housing` | Categorical | own / free / rent |
| `Saving accounts` | Ordinal | little / moderate / rich / quite rich |
| `Checking account` | Ordinal | little / moderate / rich |
| `Credit amount` | Numeric | €250–€18,424, mean €3,271 |
| `Duration` | Numeric | 4–72 months, mean 20.9 |
| `Purpose` | Categorical | 8 loan purposes (car, education, etc.) |
| `Risk` | **Target** | good (1) / bad (0) |

---

### 2.3 Exploratory Data Analysis

EDA was conducted across three axes: **univariate distributions**, **bivariate relationships with the target**, and **cross-feature correlations**. Key findings are embedded below.

---

#### Target Distribution

![Risk Class Distribution — Near-balanced after null removal (55.7% Good / 44.3% Bad), confirming class imbalance handling is still warranted](images/Risk%20distributions.png)

After removing records with missing account data, the class distribution converges toward balance. This reflects a selection effect: applicants with complete financial disclosures (both savings and checking account data) tend to cluster in a narrower risk profile, unlike the raw population which skews 70/30 toward Good.

---

#### Numerical Feature Distributions

![Distributions of Age, Credit Amount, and Duration — Credit amount and duration show right skew indicating the presence of high-value, long-tenor outlier loans](images/general%20feature%20distributions.png)

- **Age** is approximately normally distributed with a mean of 35.5 years.
- **Credit amount** is strongly right-skewed: most loans cluster below €5,000, but a long tail of high-value loans (up to €18,424) can disproportionately impact model learning.
- **Duration** mirrors the credit amount skew, with most loans between 12–36 months but a tail extending to 72 months.

---

#### Credit Amount by Risk Category

![Credit Amount Distributions by Risk Label — Bad-risk borrowers carry materially higher average loan balances (~€3,881) than good-risk borrowers (~€2,801)](images/Credit%20amount%20distributions.png)

This is one of the clearest predictive signals in the dataset. Bad-risk applicants carry an average loan balance **38.6% higher** than good-risk applicants (€3,881 vs €2,801), and a median loan duration of **25.4 months** vs **18.1 months** for good-risk applicants. Higher exposure at default combined with longer tenor is a textbook indicator of elevated credit risk.

---

#### Categorical Variables by Risk Level

![Categorical Features Split by Risk Label — Checking account and saving account status are the strongest categorical discriminators between good and bad risk](images/variables%20by%20risk%20level.png)

Account health is highly predictive. Applicants with `little` checking account balances are disproportionately concentrated in the bad-risk class, while those with `rich` or `quite rich` savings accounts skew heavily toward good-risk. This aligns with the financial intuition that liquidity buffers reduce default probability. The `Job` skill level also shows a positive relationship with creditworthiness — higher-skilled workers (Job = 3) default at lower rates.

---

#### Feature Correlation Heatmap

![Correlation Heatmap — Credit amount and duration show the strongest inter-feature correlation (r = 0.61), suggesting larger loans structurally require longer repayment windows](images/Heatamp%20for%20correlation.png)

The correlation matrix reveals a **strong positive relationship between credit amount and duration (r = 0.61)** — larger loans are naturally associated with longer repayment terms. `Job` shows moderate correlation with both duration (r = 0.20) and credit amount (r = 0.33), indicating that higher-skilled workers tend to take larger, longer-tenor loans. Age is uncorrelated with duration (r ≈ 0.00), suggesting that loan structure is not age-driven in this dataset.

---

#### Credit Amount vs. Age (by Sex and Duration)

![Scatter Plot of Credit Amount vs Age, colored by Sex, sized by Duration — No clear age-driven pattern, but larger high-risk loans (larger bubbles) appear across all age groups](images/Credit%20vs%20Age.png)

This multidimensional view confirms that **large, long-duration loans are distributed across all age groups** rather than concentrated in younger or older applicants. The largest loans (bubble size = duration) appear in the 25–50 age range regardless of sex. Mean credit amount is slightly higher for male applicants (€3,441 vs €2,937 for female), though sex alone is a weak predictor compared to account-level features.

---

### 2.4 Data Preprocessing & Feature Engineering

The preprocessing pipeline was kept intentionally lean to avoid data leakage and reflect a realistic production-deployment scenario:

**Missing Value Treatment**
- Rows with nulls in `Saving accounts` or `Checking account` were dropped (listwise deletion). Imputation was evaluated but rejected — these fields are strong predictors; imputing them risks injecting noise into the most informative features.

**Categorical Encoding**
- `Sex`, `Housing`, `Saving accounts`, and `Checking account` were encoded using **LabelEncoder** and persisted to `.pkl` files for consistent inference in the Streamlit app.
- `Purpose` was excluded from the final feature set after EDA showed weak direct association with the target relative to account-health features.
- **Design note:** LabelEncoder is appropriate here for ordinal-like features (account balance tiers); OHE was considered for nominal features but omitted to keep feature space compact given the 522-sample dataset.

**Train / Test Split**
- 80/20 stratified split (`random_state=42`) to preserve class proportions in both sets: 417 training samples, 105 test samples.

**Class Imbalance Handling**
- All tree-based models trained with `class_weight='balanced'`.
- XGBoost used `scale_pos_weight = n_negative / n_positive` (≈0.79 on training data) to up-weight the minority bad-risk class.

**Final feature set (8 features):** `Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`

---

### 2.5 Multi-Model Training & Hyperparameter Tuning

Four classifiers were benchmarked using a unified `GridSearchCV` helper (5-fold cross-validation, `scoring='accuracy'`, `n_jobs=-1`):

```python
def train_model(model, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    return best_model, accuracy_score(y_test, y_pred), grid.best_params_
```

**Decision Tree** — Searched over `max_depth` [3, 5, 7, 10, 12, None], `min_samples_split` [2, 5, 10, 12], `min_samples_leaf` [1, 2, 4, 6].

**Random Forest** — Additionally searched `n_estimators` [100, 200].

**Extra Trees** — Same grid as Random Forest.

**XGBoost** — Searched `n_estimators` [100, 200], `max_depth` [3, 5, 7, 10, 12, 15], `learning_rate` [0.01, 0.1, 0.2, 0.3], `subsample` [0.7, 1.0], `colsample_bytree` [0.7, 1.0].

---

### 2.6 Model Evaluation & Comparison

| Model | Test Accuracy | Best Parameters |
|---|---|---|
| Decision Tree | 60.0% | `max_depth=7`, `min_samples_leaf=2`, `min_samples_split=12` |
| Extra Trees | 62.9% | `max_depth=None`, `min_samples_leaf=1`, `min_samples_split=10`, `n_estimators=100` |
| Random Forest | 64.8% | `max_depth=10`, `min_samples_leaf=2`, `min_samples_split=12`, `n_estimators=100` |
| **XGBoost** | **66.7%** | `colsample_bytree=0.7`, `learning_rate=0.3`, `max_depth=3`, `n_estimators=200`, `subsample=0.7` |

**XGBoost emerged as the best-performing model** at 66.7% test accuracy, outperforming Random Forest by ~1.9 percentage points and the Decision Tree baseline by ~6.7 points.

The XGBoost configuration reflects sound credit risk modeling practice:
- **Shallow trees (`max_depth=3`)** reduce overfitting on this relatively small dataset while preserving interpretability.
- **Aggressive subsampling (`subsample=0.7`, `colsample_bytree=0.7`)** provides regularization through stochastic gradient boosting.
- **High learning rate (`0.3`) + more trees (`200`)** compensates with faster, deeper ensemble learning.

**Business interpretation:** On a 105-applicant test set, XGBoost correctly classified ~70 applicants. In a real portfolio, even a 5% lift in bad-loan detection rate translates directly to reduced expected credit losses — particularly valuable when the average bad loan balance in this dataset is €3,881.

---

### 2.7 Deployment — Streamlit Prediction App

The trained XGBoost model and label encoders are persisted via `joblib` and loaded at runtime by `app.py`, a Streamlit web application that enables real-time, single-applicant credit risk scoring.

#### Running the App Locally

```bash
# 1. Clone the repo
git clone <repo-url>
cd "Credit Risk Modeling with ML"

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit

# 4. Generate model artifacts (if not present)
jupyter notebook "EDA and Modeling.ipynb"
# Uncomment the joblib.dump() lines and run all cells

# 5. Launch the app
streamlit run app.py
# Navigate to http://localhost:8501
```

#### App Inputs & Output

| Input | Type | Range / Options |
|---|---|---|
| Age | Numeric | 18–80 |
| Sex | Dropdown | male, female |
| Job | Numeric | 0–3 (skill level) |
| Housing | Dropdown | own, rent, free |
| Saving accounts | Dropdown | little, moderate, rich, quite rich |
| Checking account | Dropdown | little, moderate, rich |
| Credit amount | Numeric | €0+ |
| Duration | Numeric | 1–72 months |

The app encodes categorical inputs using stored label encoders, constructs a single-row `DataFrame` matching the training feature schema, and returns either a **Good** or **Bad** credit risk prediction.

---

## 3. Repository Structure

```
Credit Risk Modeling with ML/
├── EDA and Modeling.ipynb      # Full analysis and model training notebook
├── app.py                      # Streamlit prediction app
├── german_credit_data.csv      # Source dataset (Kaggle German Credit Risk)
├── xgb_credit_model.pkl        # Trained XGBoost model (generated by notebook)
├── Sex_encoder.pkl             # Label encoder artifacts (generated by notebook)
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
├── target_encoder.pkl
└── images/                     # EDA and modeling visualizations
    ├── Risk distributions.png
    ├── general feature distributions.png
    ├── Credit amount distributions.png
    ├── variables by risk level.png
    ├── Heatamp for correlation.png
    └── Credit vs Age.png
```

> **Note:** The `.pkl` model and encoder files are not committed to the repository. Run all cells in `EDA and Modeling.ipynb` (uncommenting the `joblib.dump()` lines) to regenerate them before launching the app.

---

## 4. Quick Start

**Prerequisites:** Python 3.8+, pip

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit jupyter
```

**Run the notebook:**
```bash
jupyter notebook "EDA and Modeling.ipynb"
```

**Run the app:**
```bash
streamlit run app.py
```

---

## 5. Reproducibility & Extensions

**Reproducibility:**  
Set `random_state=42` is used consistently across train/test splits and all sklearn-based models. XGBoost uses `random_state=1`. Re-running the notebook will produce identical results.

**Potential Extensions:**

- **Imputation strategies:** Replace listwise deletion with median/mode imputation or a `IterativeImputer` to retain the full 1,000-row dataset and evaluate the impact on model performance.
- **Feature engineering:** Add interaction terms (e.g., `credit_amount / duration` as a monthly repayment proxy), bin `Age` into risk tiers, or encode `Purpose` with target encoding.
- **Advanced calibration:** Apply Platt scaling or isotonic regression to convert raw XGBoost scores into well-calibrated default probabilities suitable for Expected Loss calculations (EL = PD × LGD × EAD).
- **Threshold optimization:** Move beyond accuracy; optimize the decision threshold using a business cost matrix (false negative cost >> false positive cost in credit risk).
- **SHAP explainability:** Add `shap.TreeExplainer` for per-prediction feature attribution — a regulatory requirement for model risk management in many jurisdictions.
- **Containerization:** Dockerize `app.py` for cloud deployment (Streamlit Cloud, AWS ECS, Azure Container Apps).

---

## 6. License & Acknowledgements

- **Dataset:** German Credit Risk data, originally sourced from the UCI Machine Learning Repository and available on Kaggle. Please refer to the original source for licensing and citation requirements.
- **Project:** Developed by **Brandon Ytuarte / BMY Analytics** for portfolio demonstration purposes. Not intended as a production credit decision system.
- This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
