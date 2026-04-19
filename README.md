## Credit Risk Modeling with Machine Learning

This project builds and serves a **credit risk prediction model** on the classic German Credit dataset. It includes:

- **Exploratory Data Analysis (EDA)** and model development in a Jupyter notebook.
- A trained **XGBoost (or similar) classification model** saved to disk.
- A **Streamlit web app** (`app.py`) that lets users enter applicant information and instantly predict whether the credit risk is **Good (1)** or **Bad (0)**.

The goal is to demonstrate an end‑to‑end applied ML workflow for credit risk scoring: from raw CSV data to an interactive prediction tool.

---

### Project Structure

- **`german_credit_data.csv`**: Source dataset used for EDA and modeling.
- **`EDA and Modeling.ipynb`**: Notebook for data exploration, feature engineering, model training, and evaluation.
- **`app.py`**: Streamlit app that loads the trained model and encoders, collects user inputs, and outputs a credit risk prediction.
- **`xgb_credit_model.pkl`**: Trained credit risk model (loaded by `app.py`, expected to be in the project root).
- **`Sex_encoder.pkl` / `Housing_encoder.pkl` / `Saving accounts_encoder.pkl` / `Checking account_encoder.pkl`**: Label encoders for categorical features (expected to be in the project root and loaded by `app.py` via `joblib`).

> **Note**: If the model or encoder files are not present, you will need to (re)run the notebook to train the model and save these artifacts (see below).

---

### Data Description

The project uses the **German Credit Data** dataset, a tabular dataset commonly used for credit risk modeling. Each row represents a loan applicant and includes:

- **Demographic and financial features** such as:
  - `Age`
  - `Sex` (e.g. `male`, `female`)
  - `Job` (integer code, 0–3)
  - `Housing` (e.g. `own`, `rent`, `free`)
  - `Saving accounts` (e.g. `little`, `moderate`, `rich`, `quite rich`)
  - `Checking account` (e.g. `little`, `moderate`, `rich`)
  - `Credit amount`
  - `Duration` (months)
- **Target / label**: a binary indicator of **credit risk**:
  - `1` = **Good** (lower risk)
  - `0` = **Bad** (higher risk)

Exact feature cleaning, encoding, and transformations are detailed in the notebook.

---

### Environment & Requirements

You can manage dependencies via `pip` in a virtual environment. A typical environment will need:

- **Python** 3.8+ (3.9 or 3.10 recommended)
- **Core scientific stack**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
- **Modeling**:
  - `scikit-learn`
  - `xgboost` (or another gradient boosting library used in the notebook)
  - `joblib`
- **App**:
  - `streamlit`

Example installation (from the project root):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit
```

If you use `conda`, create an environment with similar packages and versions.

---

### EDA and Model Training

The **`EDA and Modeling.ipynb`** notebook walks through:

- **Loading and inspecting** `german_credit_data.csv`
- **EDA**: distributions, correlations, class balance
- **Preprocessing**:
  - Handling missing values
  - Encoding categorical variables (e.g. `Sex`, `Housing`, `Saving accounts`, `Checking account`) using label encoders
  - Scaling or transforming numerical features as needed
- **Modeling**:
  - Training a classification model to predict credit risk
  - Evaluating performance with metrics such as accuracy, confusion matrix, ROC‑AUC, etc.
- **Model persistence**:
  - Saving the trained model as `xgb_credit_model.pkl`
  - Saving label encoders as separate `*.pkl` files (one per categorical column)

To run the notebook:

1. Make sure you have the environment set up and `german_credit_data.csv` in the project root.
2. Start Jupyter (or VS Code / another notebook environment) from the project root:

   ```bash
   jupyter notebook
   ```

3. Open **`EDA and Modeling.ipynb`** and run all cells.
4. Confirm that the following files are created in the project root:
   - `xgb_credit_model.pkl`
   - `Sex_encoder.pkl`
   - `Housing_encoder.pkl`
   - `Saving accounts_encoder.pkl`
   - `Checking account_encoder.pkl`

These artifacts are what the Streamlit app uses for inference.

---

### Key Code Snippets

- **Load and preview the data**:

```python
import pandas as pd

df = pd.read_csv("german_credit_data.csv")
print(df.info())
df.head()
```

- **Basic numerical EDA (example for `Age`)**:

```python
df["Age"].describe()
```

- **Handle missing values and encode categorical features** (pattern similar to):

```python
from sklearn.preprocessing import LabelEncoder
import joblib

cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna("missing")
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f"{col}_encoder.pkl")
```

- **Train and save the model** (example with XGBoost/sklearn API):

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

X = df[["Age", "Sex", "Job", "Housing", "Saving accounts",
        "Checking account", "Credit amount", "Duration"]]
y = (df["Risk"] == "good").astype(int)  # 1 = good, 0 = bad

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "xgb_credit_model.pkl")
```

---

### EDA Visualizations (Highlights)

The notebook includes a variety of visual analyses built with `matplotlib` and `seaborn`. Some representative examples:

- **Distributions of numerical features** (Age, Credit amount, Duration, etc.):

```python
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = ["Age", "Credit amount", "Duration"]
plt.figure(figsize=(12, 4 * len(num_cols)))

for i, col in enumerate(num_cols):
    plt.subplot(len(num_cols), 1, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()
```

- **Boxplots to inspect outliers in numeric features**:

```python
plt.figure(figsize=(12, 4))

for i, col in enumerate(num_cols):
    plt.subplot(1, len(num_cols), i + 1)
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(col)

plt.tight_layout()
plt.show()
```

- **Categorical variables by risk level** (clear view of how categories split into Good vs Bad):

```python
cat_cols = ["Sex", "Housing", "Saving accounts",
            "Checking account", "Job", "Purpose"]

plt.figure(figsize=(15, 10))

for i, col in enumerate(cat_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data=df, x=col, hue="Risk")
    plt.title(f"{col} by Risk level")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

This visualization lets you quickly see, for example, how the proportion of **Good** vs **Bad** credit risks changes across categories such as `Housing` type, `Saving accounts` level, or `Checking account` status.

- **Correlation heatmap for numerical features**:

```python
num_df = df[["Age", "Job", "Credit amount", "Duration"]]
corr = num_df.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()
```

- **Example relationship plots** (e.g., Credit amount vs Age, colored by Sex or sized by Duration):

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Age",
    y="Credit amount",
    hue="Sex",
    size="Duration",
    alpha=0.7
)
plt.title("Credit amount vs Age (colored by Sex, sized by Duration)")
plt.show()
```

---

### Streamlit App: Credit Risk Prediction

The **Streamlit app** in `app.py` provides a simple web UI to make predictions using the trained model.

#### Inputs

The app collects:

- `Age` (18–80)
- `Sex` (`male`, `female`)
- `Job` (integer 0–3)
- `Housing` (`own`, `rent`, `free`)
- `Saving accounts` (`little`, `moderate`, `rich`, `quite rich`)
- `Checking account` (`little`, `moderate`, `rich`)
- `Credit amount`
- `Duration` (months)

Categorical inputs are transformed via the stored label encoders so they match the representation used during training.

#### Output

When you click **“Predict Risk”**, the app:

1. Builds a single‑row `pandas.DataFrame` with the encoded features.
2. Passes it to the loaded model (`xgb_credit_model.pkl`).
3. Displays the predicted credit risk:
   - **Good** (success message) if the model predicts `1`.
   - **Bad** (error message) if the model predicts `0`.

Core prediction logic in `app.py` (simplified):

```python
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_credit_model.pkl")
encoder = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_account = st.selectbox(
    "Saving accounts", ["little", "moderate", "rich", "quite rich"]
)
checking_account = st.selectbox(
    "Checking account", ["little", "moderate", "rich"]
)
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame(
    {
        "Age": [age],
        "Sex": [encoder["Sex"].transform([sex])[0]],
        "Job": [job],
        "Housing": [encoder["Housing"].transform([housing])[0]],
        "Saving accounts": [
            encoder["Saving accounts"].transform([saving_account])[0]
        ],
        "Checking account": [
            encoder["Checking account"].transform([checking_account])[0]
        ],
        "Credit amount": [credit_amount],
        "Duration": [duration],
    }
)

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success("The predicted credit risk is: **Good**")
    else:
        st.error("The predicted credit risk is: **Bad**")
```

---

### How to Run the App Locally

1. **Clone or download** this repository and navigate into the project directory:

   ```bash
   cd "Credit Risk Modeling with ML"
   ```

2. **Create and activate** your virtual environment (see the Environment & Requirements section) and install the required packages.

3. **Make sure model artifacts exist** in the project root:
   - `xgb_credit_model.pkl`
   - `Sex_encoder.pkl`
   - `Housing_encoder.pkl`
   - `Saving accounts_encoder.pkl`
   - `Checking account_encoder.pkl`

   If they are missing, open and run `EDA and Modeling.ipynb` to retrain and save them.

4. **Launch the Streamlit app** from the project root:

   ```bash
   streamlit run app.py
   ```

5. A browser window should open automatically (or you can navigate to the URL shown in the terminal, typically `http://localhost:8501`) to interact with the credit risk prediction UI.

---

### Reproducibility & Experimentation

- **Random seeds**: If you care about exact reproducibility of model results, set random seeds in the notebook for libraries like NumPy, scikit‑learn, and XGBoost.
- **Hyperparameters**: You can modify hyperparameters (e.g. learning rate, depth, regularization) in the notebook to experiment with performance–complexity trade‑offs.
- **Feature engineering**: Try alternative encodings, binning of continuous variables, or additional domain‑specific features to improve model quality.

---

### Potential Extensions

Some ideas to extend this project:

- **Model comparison**: Benchmark logistic regression, random forests, gradient boosting, and XGBoost.
- **Calibration**: Explore probability calibration (Platt scaling, isotonic regression) for better risk scores.
- **Explainability**: Add SHAP or feature importance plots to explain why a prediction is “Good” or “Bad”.
- **Validation**: Use cross‑validation, time‑based splits (if applicable), and robust evaluation metrics.
- **Deployment**: Containerize the app with Docker or deploy to a cloud platform (e.g. Streamlit Cloud, Heroku alternatives, etc.).

---

### License & Acknowledgements

- The **German Credit Data** dataset is widely used for educational and research purposes; please check its original source for licensing and citation details.
- This project is intended for **learning and demonstration** only and should not be used as a production‑grade credit decision system.

