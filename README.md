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


MIT License — free to use with attribution.
