import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0D1B2A;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F4F9F9;
        border-left: 5px solid #1B998B;
        padding: 1rem 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .churn-yes {
        background: #FFF0F0;
        border-left: 5px solid #FF6B6B;
        padding: 1.2rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #CC0000;
    }
    .churn-no {
        background: #F0FFF8;
        border-left: 5px solid #1B998B;
        padding: 1.2rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #0B6E5A;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    artifact = joblib.load("model.pkl")
    return artifact["model"], artifact["encoders"], artifact["features"]

model, encoders, feature_names = load_model()

# ── Header ───────────────────────────────────────────────
st.markdown('<div class="main-header">🔄 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Random Forest ML Model · B.Tech Gen AI Project · Streamlit Deployment</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar — user inputs ────────────────────────────────
with st.sidebar:
    st.header("📋 Customer Details")
    st.caption("Fill in the customer profile below")

    age = st.slider("Age", min_value=18, max_value=70, value=34, step=1)

    frequent_flyer = st.selectbox("Frequent Flyer?", options=["No", "Yes"])

    income_class = st.selectbox(
        "Annual Income Class",
        options=["Low Income", "Middle Income", "High Income"]
    )

    services_opted = st.slider("Services Opted", min_value=1, max_value=8, value=4, step=1)

    social_media = st.selectbox("Account Synced to Social Media?", options=["No", "Yes"])

    booked_hotel = st.selectbox("Booked Hotel?", options=["No", "Yes"])

    st.divider()
    predict_btn = st.button("🔍 Predict Churn", use_container_width=True, type="primary")

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

# ── Main layout ──────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.subheader("📝 Customer Profile Summary")
    profile_data = {
        "Field": ["Age", "Frequent Flyer", "Annual Income Class",
                  "Services Opted", "Social Media Sync", "Hotel Booked"],
        "Value": [age, frequent_flyer, income_class,
                  services_opted, social_media, booked_hotel]
    }
    st.dataframe(pd.DataFrame(profile_data), hide_index=True, use_container_width=True)

    # Feature importance chart
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ['#FF6B6B' if i == feat_df.index[-1] else '#1B998B' if i == feat_df.index[-2]
              else '#64748B' for i in feat_df.index]
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor='white')
    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest Feature Importance", fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("🎯 Churn Prediction")

    if predict_btn:
        input_df = prepare_input()
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        churn_prob = probability[1] * 100
        retain_prob = probability[0] * 100

        # Result box
        if prediction == 1:
            st.markdown(
                f'<div class="churn-yes">⚠️ HIGH CHURN RISK<br>'
                f'<span style="font-size:0.9rem;font-weight:400">This customer is likely to churn.</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="churn-no">✅ LOW CHURN RISK<br>'
                f'<span style="font-size:0.9rem;font-weight:400">This customer is likely to stay.</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("#### Confidence Scores")
        c1, c2 = st.columns(2)
        c1.metric("🔴 Churn Probability", f"{churn_prob:.1f}%")
        c2.metric("🟢 Retain Probability", f"{retain_prob:.1f}%")

        # Probability gauge chart
        fig2, ax2 = plt.subplots(figsize=(5, 2.5))
        bar_data = [retain_prob, churn_prob]
        bar_labels = ['Retained', 'Churned']
        bar_colors = ['#1B998B', '#FF6B6B']
        bars2 = ax2.barh(bar_labels, bar_data, color=bar_colors, edgecolor='white', height=0.5)
        for bar, val in zip(bars2, bar_data):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontweight='bold')
        ax2.set_xlim(0, 110)
        ax2.set_xlabel("Probability (%)")
        ax2.set_title("Prediction Confidence", fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Recommendation
        st.markdown("#### 💡 Recommended Action")
        if prediction == 1:
            if churn_prob >= 75:
                st.error("🚨 **Urgent:** Contact customer immediately with a personalized retention offer — discount, loyalty points, or dedicated support.")
            elif churn_prob >= 50:
                st.warning("⚡ **Moderate Risk:** Send targeted email campaign with exclusive benefits. Consider upsell to premium services.")
            else:
                st.info("🔔 **Monitor:** Flag for periodic check-in. Offer optional service bundle upgrade.")
        else:
            st.success("✅ **Healthy:** Customer appears satisfied. Consider cross-sell opportunities or referral program enrollment.")
    else:
        st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn** to get results.")

        st.markdown("#### About This Model")
        col_a, col_b = st.columns(2)
        col_a.metric("Algorithm", "Random Forest")
        col_b.metric("Accuracy", "89.01%")
        col_a.metric("AUC Score", "0.96")
        col_b.metric("Training Records", "763")

# ── Footer ───────────────────────────────────────────────
st.divider()
st.caption("🎓 B.Tech Gen AI — Final Project · Customer Churn Prediction using Random Forest · Streamlit Cloud Deployment")
