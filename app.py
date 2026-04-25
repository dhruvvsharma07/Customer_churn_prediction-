import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    artifact = joblib.load("model.pkl")
    return artifact

artifact = load_model()
model = artifact["model"]
encoders = artifact["encoders"]
features = artifact["features"]
X_test = artifact["X_test"]
y_test = artifact["y_test"]

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")
    if "Churn" not in df.columns:
        df.rename(columns={"Target": "Churn"}, inplace=True)
    return df

df = load_data()

# ---------------- STYLE ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}
.metric-card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📋 Customer Input")

age = st.sidebar.slider("Age", 18, 80, 30)
frequent_flyer = st.sidebar.selectbox("Frequent Flyer", ["Yes", "No"])
income = st.sidebar.selectbox("Income Class", ["Low Income", "Middle Income", "High Income"])
services = st.sidebar.slider("Services Opted", 1, 6, 3)
social = st.sidebar.selectbox("Social Media Sync", ["Yes", "No"])
hotel = st.sidebar.selectbox("Booked Hotel", ["Yes", "No"])

predict_btn = st.sidebar.button("🚀 Predict Churn")

# ---------------- HEADER ----------------
st.title("📊 Customer Churn Intelligence Dashboard")
st.caption("AI-powered retention analytics system")

# ---------------- INPUT PROCESS ----------------
def prepare_input():
    row = {
        "Age": age,
        "FrequentFlyer": frequent_flyer,
        "AnnualIncomeClass": income,
        "ServicesOpted": services,
        "AccountSyncedToSocialMedia": social,
        "BookedHotelOrNot": hotel,
    }
    df_input = pd.DataFrame([row])

    for col in encoders:
        df_input[col] = encoders[col].transform(df_input[col])

    return df_input[features]

# ---------------- MAIN ----------------
if predict_btn:

    input_df = prepare_input()
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    churn_prob = prob[1]
    retain_prob = prob[0]

    # ---------------- KPI CARDS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Churn Risk", f"{churn_prob*100:.2f}%")
    col2.metric("Retention Chance", f"{retain_prob*100:.2f}%")

    if prediction == 1:
        col3.error("⚠️ HIGH RISK")
    else:
        col3.success("✅ LOW RISK")

    st.divider()

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📊 Insights", "📈 Model Performance", "⚙️ Feature Analysis"])

    # -------- INSIGHTS --------
    with tab1:
        st.subheader("Customer Data Insights")

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(4,3))
            df['Churn'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title("Churn Distribution")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.countplot(data=df, x='ServicesOpted', hue='Churn', ax=ax)
            ax.set_title("Services vs Churn")
            st.pyplot(fig)

        c3, c4 = st.columns(2)

        with c3:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.countplot(data=df, x='AnnualIncomeClass', hue='Churn', ax=ax)
            plt.xticks(rotation=15)
            ax.set_title("Income vs Churn")
            st.pyplot(fig)

        with c4:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.histplot(data=df, x='Age', hue='Churn', kde=True, ax=ax)
            ax.set_title("Age Distribution")
            st.pyplot(fig)

    # -------- MODEL PERFORMANCE --------
    with tab2:
        st.subheader("Model Evaluation")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        c1, c2 = st.columns(2)

        # Confusion Matrix
        with c1:
            fig_cm, ax_cm = plt.subplots(figsize=(4,3))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

        # ROC Curve
        with c2:
            fig_roc, ax_roc = plt.subplots(figsize=(4,3))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax_roc.plot([0,1],[0,1],'--')
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

    # -------- FEATURE IMPORTANCE --------
    with tab3:
        st.subheader("Feature Importance")

        importances = model.feature_importances_

        fig_fi, ax_fi = plt.subplots(figsize=(5,3))
        sns.barplot(x=importances, y=features, ax=ax_fi)
        ax_fi.set_title("Feature Importance")
        st.pyplot(fig_fi)

    # ---------------- RECOMMENDATION ----------------
    st.divider()
    st.subheader("💡 Business Recommendation")

    if prediction == 1:
        st.warning("Offer discounts, loyalty rewards, or personalized outreach to retain this customer.")
    else:
        st.success("Customer is stable. Consider upselling premium features or referrals.")

# ---------------- EMPTY STATE ----------------
else:
    st.info("👈 Enter customer details in the sidebar and click 'Predict Churn' to view insights.")