import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------- LOAD ----------------------
artifact = joblib.load("model.pkl")
model = artifact["model"]
encoders = artifact["encoders"]
features = artifact["features"]

df = pd.read_csv("customer_churn.csv")

# ---------------------- STYLE ----------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("📋 Customer Input")

age = st.sidebar.slider("Age", 18, 80, 30)
frequent_flyer = st.sidebar.selectbox("Frequent Flyer", ["Yes", "No"])
income = st.sidebar.selectbox("Income Class", ["Low Income", "Middle Income", "High Income"])
services = st.sidebar.slider("Services Opted", 1, 6, 3)
social = st.sidebar.selectbox("Social Media Sync", ["Yes", "No"])
hotel = st.sidebar.selectbox("Booked Hotel", ["Yes", "No"])

predict_btn = st.sidebar.button("🚀 Predict Churn")

# ---------------------- HEADER ----------------------
st.title("📊 Customer Churn Intelligence Dashboard")
st.caption("AI-powered retention analytics system")

# ---------------------- INPUT DF ----------------------
input_dict = {
    "Age": age,
    "FrequentFlyer": frequent_flyer,
    "AnnualIncomeClass": income,
    "ServicesOpted": services,
    "AccountSyncedToSocialMedia": social,
    "BookedHotelOrNot": hotel
}

input_df = pd.DataFrame([input_dict])

# Encode
for col in encoders:
    input_df[col] = encoders[col].transform(input_df[col])

# ---------------------- MAIN LOGIC ----------------------
if predict_btn:

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    churn_prob = prob[1]
    retain_prob = prob[0]

    # ---------------------- METRICS ----------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Risk", f"{churn_prob*100:.2f}%")

    with col2:
        st.metric("Retention Chance", f"{retain_prob*100:.2f}%")

    with col3:
        if prediction == 1:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("✅ LOW RISK")

    st.divider()

    # ---------------------- TABS ----------------------
    tab1, tab2, tab3 = st.tabs(["📊 Insights", "📈 Model Performance", "⚙️ Feature Analysis"])

    # ===================== TAB 1 =====================
    with tab1:
        st.subheader("Customer Data Insights")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(4,3))
            df['Target'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title("Churn Distribution")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.countplot(data=df, x='ServicesOpted', hue='Target', ax=ax)
            ax.set_title("Services vs Churn")
            st.pyplot(fig)

        col3, col4 = st.columns(2)

        with col3:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.countplot(data=df, x='AnnualIncomeClass', hue='Target', ax=ax)
            plt.xticks(rotation=15)
            ax.set_title("Income vs Churn")
            st.pyplot(fig)

        with col4:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.histplot(data=df, x='Age', hue='Target', kde=True, ax=ax)
            ax.set_title("Age Distribution")
            st.pyplot(fig)

    # ===================== TAB 2 =====================
    with tab2:
        st.subheader("Model Evaluation")

        col1, col2 = st.columns(2)

        with col1:
            st.image("confusion_matrix.png", caption="Confusion Matrix")

        with col2:
            st.image("roc_curve.png", caption="ROC Curve")

    # ===================== TAB 3 =====================
    with tab3:
        st.subheader("Feature Importance")

        st.image("feature_importance.png")

        st.subheader("Correlation Heatmap")
        st.image("correlation_heatmap.png")

    # ---------------------- RECOMMENDATION ----------------------
    st.divider()

    st.subheader("💡 Recommendation")

    if prediction == 1:
        st.warning("Offer discounts, loyalty benefits, and targeted engagement.")
    else:
        st.success("Customer is stable. Consider upselling premium services.")

# ---------------------- EMPTY STATE ----------------------
else:
    st.info("👈 Enter customer details and click 'Predict Churn' to view analytics.")