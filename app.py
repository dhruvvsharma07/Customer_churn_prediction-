import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD ----------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

artifact = load_model()
model = artifact["model"]
encoders = artifact["encoders"]
features = artifact["features"]
X_test = artifact["X_test"]
y_test = artifact["y_test"]

@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")
    if "Churn" not in df.columns:
        df.rename(columns={"Target": "Churn"}, inplace=True)
    return df

df = load_data()

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

[data-testid="stSidebar"] {
    background: #020617;
}

.block-container {
    padding-top: 1.5rem;
}

/* Glass Cards */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    color: white;
    border-radius: 10px;
    font-weight: bold;
    height: 3em;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
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

predict_btn = st.sidebar.button("🚀 Predict")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center; font-size:48px;'>📊 Churn Intelligence</h1>
<p style='text-align:center; color:gray;'>AI-powered retention analytics</p>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
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
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    churn_prob = prob[1]
    retain_prob = prob[0]

    # ---------------- KPI ----------------
    c1, c2, c3 = st.columns(3)

    c1.metric("Churn Risk", f"{churn_prob*100:.1f}%")
    c2.metric("Retention", f"{retain_prob*100:.1f}%")

    if pred == 1:
        c3.error("⚠️ HIGH RISK")
    else:
        c3.success("✅ STABLE")

    st.markdown("---")

    # ---------------- GAUGE ----------------
    st.subheader("🎯 Risk Gauge")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        title={'text': "Churn Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ]
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📊 Insights", "📈 Performance", "⚙️ Features"])

    # -------- INSIGHTS --------
    with tab1:
        st.subheader("Data Insights")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x="Age", color="Churn")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df, x="ServicesOpted", color="Churn")
            st.plotly_chart(fig, use_container_width=True)

    # -------- PERFORMANCE --------
    with tab2:
        st.subheader("Model Performance")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        col1, col2 = st.columns(2)

        # ---------------- CONFUSION MATRIX ----------------
        with col1:
            cm = confusion_matrix(y_test, y_pred)

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Retained", "Churned"],
                y=["Retained", "Churned"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}"
            ))

            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )

            st.plotly_chart(fig_cm, use_container_width=True)

        # ---------------- ROC CURVE ----------------
        with col2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"AUC = {roc_auc:.2f}"
            ))

            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1],
                mode='lines',
                line=dict(dash='dash'),
                name="Random"
            ))

            fig_roc.update_layout(title="ROC Curve")

            st.plotly_chart(fig_roc, use_container_width=True)

    # -------- FEATURES --------
    with tab3:
        st.subheader("Feature Importance")

        importances = model.feature_importances_

        fig = px.bar(
            x=importances,
            y=features,
            orientation='h'
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- RECOMMEND ----------------
    st.markdown("---")
    st.subheader("💡 Recommendation")

    if pred == 1:
        st.warning("Target this user with retention campaigns and discounts.")
    else:
        st.success("Upsell premium services or loyalty programs.")

else:
    st.markdown("""
    <div style='text-align:center; padding: 80px;'>
        <h2>🚀 Ready to Predict</h2>
        <p>Enter customer details in the sidebar</p>
    </div>
    """, unsafe_allow_html=True)