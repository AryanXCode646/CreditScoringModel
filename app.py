"""Streamlit UI for educational Credit Scoring Model inference."""

from __future__ import annotations

import streamlit as st

from credit_scoring.data import CATEGORY_LABELS
from credit_scoring.predict import DEFAULT_SAMPLE_RECORD, load_model_bundle, predict_credit_risk

st.set_page_config(page_title="Credit Scoring Model — CodeAlpha", layout="centered")
st.title("Credit Scoring Model (Educational Demo)")
st.warning(
    "Educational ML Demonstration Only: This tool uses a model trained on the 1994 UCI "
    "Statlog German Credit dataset and must NOT be used for real lending or financial decisions."
)


@st.cache_resource
def _get_bundle():
    return load_model_bundle("models/credit_scoring_bundle.joblib")


try:
    bundle = _get_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


def _select(label: str, feature: str) -> str:
    options = CATEGORY_LABELS[feature]
    codes = list(options.keys())
    default = DEFAULT_SAMPLE_RECORD[feature]
    return st.selectbox(
        label,
        options=codes,
        index=codes.index(default),
        format_func=lambda code: f"{code} — {options[code]}",
    )


with st.form("credit_form"):
    col1, col2 = st.columns(2)
    with col1:
        duration_months = st.number_input("Duration (months)", min_value=1, max_value=120, value=24)
        credit_amount = st.number_input(
            "Credit Amount (DM)", min_value=100, max_value=50000, value=4500
        )
        installment_commitment = st.slider("Installment Rate (% of income)", 1, 4, 3)
        residence_since = st.slider("Residence Since (years)", 1, 4, 2)
        age = st.number_input("Age", min_value=18, max_value=100, value=32)
        existing_credits = st.slider("Existing Credits at Bank", 1, 4, 1)
        num_dependents = st.slider("Number of Dependents", 1, 2, 1)
        checking_status = _select("Checking Account", "checking_status")
        credit_history = _select("Credit History", "credit_history")
        purpose = _select("Loan Purpose", "purpose")
    with col2:
        savings_status = _select("Savings Account", "savings_status")
        employment = _select("Employment Duration", "employment")
        personal_status_sex = _select("Personal Status", "personal_status_sex")
        other_parties = _select("Other Debtors/Guarantors", "other_parties")
        property_magnitude = _select("Property", "property_magnitude")
        other_payment_plans = _select("Other Payment Plans", "other_payment_plans")
        housing = _select("Housing", "housing")
        job = _select("Job Category", "job")
        own_telephone = _select("Telephone", "own_telephone")
        foreign_worker = _select("Foreign Worker", "foreign_worker")

    submitted = st.form_submit_button("Estimate Credit Risk")

if submitted:
    record = {
        "checking_status": checking_status,
        "duration_months": duration_months,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment": employment,
        "installment_commitment": installment_commitment,
        "personal_status_sex": personal_status_sex,
        "other_parties": other_parties,
        "residence_since": residence_since,
        "property_magnitude": property_magnitude,
        "age": age,
        "other_payment_plans": other_payment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "own_telephone": own_telephone,
        "foreign_worker": foreign_worker,
    }
    res = predict_credit_risk(record, bundle)[0]
    st.subheader(f"Prediction: {res['predicted_label']}")
    st.metric("Estimated High-Risk Probability", f"{res['probability_high_risk']:.1%}")
    st.caption(f"Model: `{res['model_name']}` | Decision threshold: `{res['threshold_used']:.2f}`")
