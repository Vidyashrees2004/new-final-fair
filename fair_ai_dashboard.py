import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import time
import psutil

from lightgbm import LGBMClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference
from codecarbon import EmissionsTracker

st.set_page_config(page_title="Fair AI Dashboard", layout="wide")

st.title("🎯 Fair AI Income Prediction")
st.markdown("Baseline vs Fair Model • Explainability • Energy Tracking")

# ======================================
# Load baseline + scaler
# ======================================

@st.cache_resource
def load_assets():
    baseline = joblib.load("models/baseline_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/feature_names.pkl")
    return baseline, scaler, features

baseline_model, scaler, features = load_assets()

# ======================================
# Re-Train Fair Model at Runtime
# ======================================

@st.cache_resource
def train_fair_model():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, header=None, na_values=" ?", skipinitialspace=True)

    df.columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'
    ]

    df = df.dropna()

    df_fixed = pd.DataFrame()
    df_fixed['age'] = df['age']
    df_fixed['education-num'] = df['education-num']
    df_fixed['hours-per-week'] = df['hours-per-week']
    df_fixed['sex'] = (df['sex'] == 'Male').astype(int)
    df_fixed['race'] = (df['race'] == 'White').astype(int)
    df_fixed['income_binary'] = (df['income'] == '>50K').astype(int)

    X = df_fixed[features]
    y = df_fixed['income_binary']
    sens = df_fixed['sex']

    X_scaled = scaler.transform(X)

    fair_model = ExponentiatedGradient(
        LGBMClassifier(n_estimators=100, random_state=42),
        constraints=DemographicParity()
    )

    fair_model.fit(X_scaled, y, sensitive_features=sens)

    return fair_model, X_scaled, y, sens

fair_model, X_full_scaled, y_full, sens_full = train_fair_model()

# ======================================
# Sidebar Inputs
# ======================================

st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 80, 35)
education = st.sidebar.slider("Education Years", 1, 16, 13)
hours = st.sidebar.slider("Hours per Week", 10, 80, 40)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
race = st.sidebar.selectbox("Race", ["Non-White", "White"])

gender_num = 1 if gender == "Male" else 0
race_num = 1 if race == "White" else 0

model_choice = st.sidebar.radio("Choose Model", ["Baseline", "Fair Model"])
run = st.sidebar.button("Run Prediction")

# ======================================
# Prediction
# ======================================

if run:

    input_data = np.array([[age, education, hours, gender_num, race_num]])
    scaled_data = scaler.transform(input_data)

    tracker = EmissionsTracker(save_to_file=False)
    tracker.start()

    start = time.time()

    if model_choice == "Baseline":
        prediction = baseline_model.predict(scaled_data)[0]
        probability = baseline_model.predict_proba(scaled_data)[0][1]
        shap_model = baseline_model
        explain_title = "🧠 Explainability"

    else:
        prediction = fair_model.predict(scaled_data)[0]
        probability = fair_model._pmf_predict(scaled_data)[0][1]

        # Use baseline for explanation (fairlearn wrapper not SHAP compatible)
        shap_model = baseline_model
        explain_title = "🧠 Explainability (Raw Model Before Fairness Adjustment)"

    inference_time = time.time() - start
    emissions = tracker.stop()

    # Result Display
    if prediction == 1:
        st.success("💰 HIGH Income (>50K)")
    else:
        st.info("📉 LOW Income (≤50K)")

    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{probability:.2%}")
    col2.metric("Inference Time (ms)", f"{inference_time*1000:.2f}")

    st.metric("CO₂ Emission (kg)", f"{emissions:.8f}")

    # ======================================
    # SHAP Explainability
    # ======================================

    st.markdown("---")
    st.subheader(explain_title)

    explainer = shap.Explainer(shap_model)
    shap_values = explainer(scaled_data)

    shap_df = pd.DataFrame({
        "Feature": features,
        "Impact": shap_values.values[0]
    }).sort_values("Impact")

    st.bar_chart(shap_df.set_index("Feature"))

    st.write("🔺 Most Positive Factor:", shap_df.iloc[-1]["Feature"])


# ======================================
# Fairness Metrics
# ======================================

st.markdown("---")
st.header("📊 Fairness Comparison")

fair_pred = fair_model.predict(X_full_scaled)
baseline_pred = baseline_model.predict(X_full_scaled)

baseline_gap = demographic_parity_difference(
    y_full, baseline_pred, sensitive_features=sens_full
)

fair_gap = demographic_parity_difference(
    y_full, fair_pred, sensitive_features=sens_full
)

colA, colB = st.columns(2)
colA.metric("Baseline Gap", f"{baseline_gap:.4f}")
colB.metric("Fair Model Gap", f"{fair_gap:.4f}")

if fair_gap < baseline_gap:
    st.success("✅ Fair model reduces demographic disparity")
