import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="SL Vehicle Price Ensemble Dashboard", page_icon="🚗", layout="wide")

# Load assets once and cache them for fast performance
@st.cache_resource
def load_assets():
    preprocessor = joblib.load('preprocessor.joblib')
    models = {
        'Hybrid (RF + ANN)': joblib.load('hybrid_model.joblib'),
        'Stacking (RF+XGB+LGBM)': joblib.load('stacking_model.joblib'),
        'Random Forest (Bagging)': joblib.load('rf_model.joblib'),
        'XGBoost (Boosting)': joblib.load('xgb_model.joblib')
    }
    results_df = pd.read_csv('model_results.csv')
    feat_df = pd.read_csv('feature_importance.csv')
    with open('ui_options.json', 'r') as f:
        ui_opts = json.load(f)
    return preprocessor, models, results_df, feat_df, ui_opts

preprocessor, models, results_df, feat_df, ui_opts = load_assets()

# Header
st.title("🚗 Sri Lankan Vehicle Price Predictor — Ensemble Learning Dashboard")
st.markdown("Comparing **Bagging (Random Forest)**, **Boosting (XGBoost)**, **Stacking**, and **Hybrid (RF + ANN)** on local Sri Lankan market data (`ikman.lk` & `riyasewana.lk`).")

tab1, tab2, tab3 = st.tabs(["🔮 Live Price Prediction", "📊 Model Comparison", "🔍 Feature Importance & Insights"])

# --- TAB 1: LIVE PREDICTION ---
with tab1:
    st.subheader("Enter Vehicle Specifications")
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", ui_opts['brands'])
        model_name = st.selectbox("Model", ui_opts['brand_models'][brand])
        yom = st.slider("Year of Manufacture (YOM)", ui_opts['yom_min'], ui_opts['yom_max'], 2017)
        condition = st.selectbox("Condition", ui_opts['conditions'])

    with col2:
        engine_cc = st.number_input("Engine Capacity (cc)", min_value=ui_opts['cc_min'], max_value=ui_opts['cc_max'], value=1500, step=100)
        mileage = st.number_input("Mileage (KM)", min_value=0, max_value=ui_opts['km_max'], value=65000, step=5000)
        gear = st.selectbox("Transmission (Gear)", ui_opts['gears'])
        fuel = st.selectbox("Fuel Type", ui_opts['fuels'])

    with col3:
        town = st.selectbox("Town / Location", ui_opts['towns'])
        leasing = st.selectbox("Leasing Status", ui_opts['leasings'])
        selected_model = st.selectbox("Primary Ensemble Model", list(models.keys()))
        ac = st.checkbox("Air Condition", value=True)
        ps = st.checkbox("Power Steering", value=True)
        pm = st.checkbox("Power Mirror", value=True)
        pw = st.checkbox("Power Window", value=True)

    if st.button("Predict Vehicle Price (LKR)", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'YOM': yom,
            'Engine (cc)': engine_cc,
            'Millage(KM)': mileage,
            'Brand': brand,
            'Model': model_name,
            'Gear': gear,
            'Fuel Type': fuel,
            'Town': town,
            'Leasing': leasing,
            'Condition': condition,
            'AIR CONDITION': 'Available' if ac else 'Not Available',
            'POWER STEERING': 'Available' if ps else 'Not Available',
            'POWER MIRROR': 'Available' if pm else 'Not Available',
            'POWER WINDOW': 'Available' if pw else 'Not Available'
        }])

        X_input = preprocessor.transform(input_df)
        pred_lakhs = models[selected_model].predict(X_input)[0]
        pred_millions = pred_lakhs / 10.0

        st.success(f"### Estimated Price ({selected_model}): **Rs. {pred_millions:,.2f} Million** ({pred_lakhs:,.2f} Lakhs)")

        # Dynamic comparison across all 4 ensemble families
        st.markdown("#### Side-by-Side Comparison Across All Ensemble Families")
        comp_cols = st.columns(4)
        for idx, (m_name, m_obj) in enumerate(models.items()):
            p_lakhs = m_obj.predict(X_input)[0]
            comp_cols[idx].metric(label=m_name, value=f"Rs. {p_lakhs/10.0:,.2f}M", delta=f"{p_lakhs:,.1f} Lakhs")

# --- TAB 2: MODEL COMPARISON ---
with tab2:
    st.subheader("Ensemble & Hybrid Model Evaluation Matrix")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**R² Score (Higher is Better)**")
        st.bar_chart(results_df.set_index('Model')['R2 Score'])
    with c2:
        st.markdown("**RMSE in LKR Lakhs (Lower is Better)**")
        st.bar_chart(results_df.set_index('Model')['RMSE (Lakhs)'])

# --- TAB 3: FEATURE IMPORTANCE & INSIGHTS ---
with tab3:
    st.subheader("Top 15 Features Driving Sri Lankan Vehicle Prices")
    st.bar_chart(feat_df.set_index('Feature')['Importance'])

    st.markdown("""
    ### Key Analytical Takeaways
    * **Why Hybrid (RF + ANN) Won (R² = 0.9366, MAE = 4.18 Lakhs):** Pairing Random Forest's sharp decision boundaries on categorical Sri Lankan brands/models with an Artificial Neural Network's smooth depreciation curves reduces both bias and variance.
    * **Stacking (R² = 0.9266):** Combining Bagging (RF) and Boosting (XGBoost, LightGBM) via a Ridge meta-learner outperformed every standalone tree model.
    * **Bagging Generalization (RF Test R² = 0.9250 vs. OOB R² = 0.9207):** Nearly identical Out-of-Bag and Test scores prove strong variance reduction without overfitting.
    """)
