# ✈️ Customer Churn Prediction — Airline Industry

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-predictor372.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live Demo
👉 **[customer-churn-predictor372.streamlit.app](https://customer-churn-predictor372.streamlit.app/)**

Fill in a customer's details → Get instant churn prediction with probability score.

---

## 📌 Problem Statement

**Customer churn** = when a customer stops using a company's service.

For airlines, predicting churn early allows targeted retention campaigns before the customer leaves. Retaining an existing customer costs **5–7× less** than acquiring a new one. Even a 5% reduction in churn can increase profits by **25–95%**.

---

## 📁 Repository Structure

```
Customer_Churn_Prediction/
│
├── Customer_Churn_Prediction.ipynb   # Full ML pipeline — EDA to evaluation
├── app.py                            # Streamlit web app (deployed)
├── customer_churn.csv                # Dataset (954 rows × 7 columns)
├── model.pkl                         # Saved RF model + encoders (joblib)
├── requirements.txt                  # Python dependencies
│
├── assets/                           # Saved plot images from notebook
│   ├── churn_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
└── README.md
```

---

## 📊 Dataset Overview

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Customer age (27–38 years) |
| `FrequentFlyer` | Categorical | Yes / No |
| `AnnualIncomeClass` | Categorical | Low / Middle / High Income |
| `ServicesOpted` | Numeric | Number of services used (1–6) |
| `AccountSyncedToSocialMedia` | Categorical | Yes / No |
| `BookedHotelOrNot` | Categorical | Yes / No |
| `Target` | Binary | 0 = Retained, 1 = Churned |

**Class Distribution:** 730 Retained (76.5%) · 224 Churned (23.5%)

---

## 🔬 ML Pipeline — Step by Step

### 1. Exploratory Data Analysis (EDA)
- Churn distribution — bar chart + pie chart
- Feature-wise churn breakdown (6 subplots)
- Key finding: customers opting for fewer services churn significantly more

### 2. Data Cleaning
- `FrequentFlyer` had `'No Record'` values → replaced with `'No'` (logically equivalent)
- Checked and confirmed zero null values — 954 rows retained

### 3. Label Encoding
- 4 categorical columns encoded using `LabelEncoder`
- Encoders saved alongside model for consistent inference in deployment

### 4. Correlation Heatmap
- Checked for multicollinearity between features
- No extreme correlations found — all 6 features retained

### 5. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 763 training samples | 191 testing samples
# stratify=y ensures equal churn ratio in both sets
```

### 6. Model — Random Forest Classifier
```python
rf_model = RandomForestClassifier(
    n_estimators=100,     # 100 decision trees
    max_depth=10,         # prevents overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1             # all CPU cores
)
```

**Why Random Forest over others?**

| Model | Why NOT chosen |
|---|---|
| Logistic Regression | Assumes linear relationships — churn is non-linear |
| Single Decision Tree | Highly prone to overfitting |
| SVM | Doesn't handle mixed data types well; no feature importance |
| Neural Network | Needs much more data; black box — no explainability |
| XGBoost | Strong alternative — planned for future comparison |

### 7. Model Persistence
```python
artifact = {
    'model': rf_model,
    'encoders': encoders,        # saved for inference consistency
    'features': list(X.columns),
    'X_test': X_test,
    'y_test': y_test
}
joblib.dump(artifact, 'model.pkl')
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~92% |
| ROC-AUC | > 0.85 |
| Recall (Churned) | High — minimizes missed churners |

### Feature Importance (Highest → Lowest)

```
1. ServicesOpted              ████████████████████  Most Important
2. FrequentFlyer              ███████████████
3. AnnualIncomeClass          ████████████
4. Age                        ██████████
5. BookedHotelOrNot           █████
6. AccountSyncedToSocialMedia ████
```

### Key Insight
> Customers opting for **fewer services** are far more likely to churn.
> Surprisingly, **FrequentFlyers churn MORE** — they have more options and switch airlines for better deals.

---

## 🚀 Deployment

**Platform:** Streamlit Community Cloud
**Flow:** Train in Jupyter → `joblib.dump()` → push to GitHub → auto-deploy on Streamlit Cloud

```
Notebook → model.pkl → GitHub → Streamlit Cloud → Live App
```

---

## 🛠️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/dhruvvsharma07/Customer_churn_prediction-
cd Customer_churn_prediction-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebook (for training)
jupyter notebook Customer_Churn_Prediction.ipynb

# 4. Run Streamlit app
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

---

## 🔮 Future Improvements

- [ ] **GridSearchCV** — systematic hyperparameter tuning
- [ ] **K-Fold Cross Validation** — more reliable accuracy estimation
- [ ] **XGBoost / LightGBM** — compare gradient boosting vs bagging
- [ ] **SMOTE** — handle class imbalance synthetically
- [ ] **SHAP Values** — instance-level explainability

---

## 👨‍💻 Author

**Dhruv Sharma**
B.Tech Gen AI | Roll No: KU2507U0372
IBM Project — 2nd Semester

---

## 📄 License
MIT License — free to use with attribution.
