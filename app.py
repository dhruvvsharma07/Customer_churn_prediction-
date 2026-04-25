import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── LOAD MODEL ─────────────────────────────
@st.cache_resource
def load_model():
    artifact = joblib.load("model.pkl")
    return (
        artifact["model"],
        artifact["encoders"],
        artifact["features"],
        artifact["X_test"],
        artifact["y_test"]
    )

model, encoders, feature_names, X_test, y_test = load_model()

# ── LOAD DATA ─────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")
    return df

df = load_data()

# 🔥 Fix column naming once
if "Churn" not in df.columns:
    df.rename(columns={"Target": "Churn"}, inplace=True)

# ── SIDEBAR ───────────────────────────────
with st.sidebar:
    st.title("📋 Customer Input")

    age = st.slider("Age", 18, 70, 34)
    frequent_flyer = st.selectbox("Frequent Flyer?", ["No", "Yes"])
    income_class = st.selectbox("Income Class", ["Low Income", "Middle Income", "High Income"])
    services_opted = st.slider("Services Opted", 1, 8, 4)
    social_media = st.selectbox("Social Media Sync?", ["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel?", ["No", "Yes"])

    predict_btn = st.button("🔮 Predict", use_container_width=True)

# ── INPUT PREP ────────────────────────────
def prepare_input():
    row = {
        "Age": age,
        "FrequentFlyer": encoders["FrequentFlyer"].transform([frequent_flyer])[0],
        "AnnualIncomeClass": encoders["AnnualIncomeClass"].transform([income_class])[0],
        "ServicesOpted": services_opted,
        "AccountSyncedToSocialMedia": encoders["AccountSyncedToSocialMedia"].transform([social_media])[0],
        "BookedHotelOrNot": encoders["BookedHotelOrNot"].transform([booked_hotel])[0],
    }
    return pd.DataFrame([row])[feature_names]

# ── HEADER ────────────────────────────────
st.title("📊 Customer Churn Analytics Dashboard")
st.caption("End-to-End ML Project · Random Forest · Streamlit")

st.divider()

# ── TOP METRICS ───────────────────────────
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
col3.metric("Model Type", "Random Forest")

st.divider()

# ── PREDICTION SECTION ────────────────────
st.subheader("🔮 Prediction")

if predict_btn:
    input_df = prepare_input()
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    churn_prob = prob[1] * 100
    retain_prob = prob[0] * 100

    colA, colB = st.columns([1,1])

    with colA:
        if prediction == 1:
            st.error(f"⚠️ High Churn Risk ({churn_prob:.1f}%)")
        else:
            st.success(f"✅ Low Churn Risk ({retain_prob:.1f}%)")

    with colB:
        fig, ax = plt.subplots(figsize=(4,2))
        ax.barh(["Retained", "Churned"], [retain_prob, churn_prob])
        ax.set_xlim(0,100)
        st.pyplot(fig)

st.divider()

# ── FEATURE IMPORTANCE ─────────────────────
st.subheader("📊 Feature Importance")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance")

fig, ax = plt.subplots(figsize=(6,3))
ax.barh(feat_df["Feature"], feat_df["Importance"])
st.pyplot(fig)

# ── EDA SECTION ───────────────────────────
st.subheader("📈 Data Insights")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(4,3))
    df['Churn'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(4,3))
    df.groupby("ServicesOpted")["Churn"].mean().plot(ax=ax)
    ax.set_title("Churn vs Services")
    st.pyplot(fig)

# ── MODEL PERFORMANCE ─────────────────────
st.subheader("📉 Model Performance")

col1, col2 = st.columns(2)

# Confusion Matrix
with col1:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4,3))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

# ROC Curve
with col2:
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax.plot([0,1],[0,1],'--')
    ax.legend()
    st.pyplot(fig)

st.success("✅ App fully loaded with analytics + prediction")