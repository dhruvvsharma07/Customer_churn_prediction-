import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────
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

# ── Load dataset ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("customer_churn.csv")

df = load_data()

# ── Header ───────────────────────────────────────────────
st.title("🔄 Customer Churn Predictor")
st.caption("Random Forest ML Model · Gen AI Project")

st.divider()

# ── Sidebar — user inputs ────────────────────────────────
with st.sidebar:
    st.header("📋 Customer Details")

    age = st.slider("Age", 18, 70, 34)
    frequent_flyer = st.selectbox("Frequent Flyer?", ["No", "Yes"])
    income_class = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])
    services_opted = st.slider("Services Opted", 1, 8, 4)
    social_media = st.selectbox("Social Media Sync?", ["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel?", ["No", "Yes"])

    predict_btn = st.button("🔍 Predict Churn")

# ── Prepare input ────────────────────────────────────────
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

# ── Tabs Layout ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Insights", "📈 Model Performance"])

# =========================================================
# 🔮 TAB 1 — PREDICTION
# =========================================================
with tab1:

    if predict_btn:
        input_df = prepare_input()
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        churn_prob = probability[1] * 100
        retain_prob = probability[0] * 100

        if prediction == 1:
            st.error("⚠️ HIGH CHURN RISK")
        else:
            st.success("✅ LOW CHURN RISK")

        st.metric("Churn Probability", f"{churn_prob:.2f}%")
        st.metric("Retain Probability", f"{retain_prob:.2f}%")

        # Probability chart
        fig, ax = plt.subplots()
        ax.barh(["Retained", "Churned"], [retain_prob, churn_prob])
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

    else:
        st.info("Enter details and click Predict")

    # Feature importance
    st.subheader("📊 Feature Importance")

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance")

    fig, ax = plt.subplots()
    ax.barh(feat_df["Feature"], feat_df["Importance"])
    ax.set_title("Feature Importance")
    st.pyplot(fig)


# =========================================================
# 📊 TAB 2 — DATA INSIGHTS (EDA)
# =========================================================
with tab2:

    st.subheader("Churn Distribution")

    fig, ax = plt.subplots()
    df['Churn'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

    st.subheader("Age Distribution")

    fig, ax = plt.subplots()
    df['Age'].hist(ax=ax, bins=20)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    st.subheader("Services Opted vs Churn")

    fig, ax = plt.subplots()
    df.groupby("ServicesOpted")["Churn"].mean().plot(ax=ax)
    ax.set_title("Churn Rate vs Services")
    st.pyplot(fig)


# =========================================================
# 📈 TAB 3 — MODEL PERFORMANCE
# =========================================================
with tab3:

    st.subheader("Confusion Matrix")

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], '--')
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    st.metric("AUC Score", f"{roc_auc:.3f}")