import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------

model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Features
# --------------------------------------------------

features = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🧠 Parkinson's Voice Analysis")

st.write(
    """
    Enter the voice biomarker measurements below to obtain a
    machine-learning classification.
    """
)

st.warning(
    "⚠️ This is an experimental research prototype and is NOT "
    "a medical diagnostic tool."
)

# --------------------------------------------------
# Input Form
# --------------------------------------------------

st.subheader("Enter Voice Biomarker Values")

with st.form("parkinsons_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        fo = st.number_input(
            "MDVP:Fo(Hz)",
            value=120.0
        )

        fhi = st.number_input(
            "MDVP:Fhi(Hz)",
            value=150.0
        )

        flo = st.number_input(
            "MDVP:Flo(Hz)",
            value=100.0
        )

        jitter = st.number_input(
            "MDVP:Jitter(%)",
            value=0.007
        )

        jitter_abs = st.number_input(
            "MDVP:Jitter(Abs)",
            value=0.00007,
            format="%.8f"
        )

        rap = st.number_input(
            "MDVP:RAP",
            value=0.003
        )

        ppq = st.number_input(
            "MDVP:PPQ",
            value=0.005
        )

        ddp = st.number_input(
            "Jitter:DDP",
            value=0.010
        )

    with col2:

        shimmer = st.number_input(
            "MDVP:Shimmer",
            value=0.04
        )

        shimmer_db = st.number_input(
            "MDVP:Shimmer(dB)",
            value=0.4
        )

        apq3 = st.number_input(
            "Shimmer:APQ3",
            value=0.02
        )

        apq5 = st.number_input(
            "Shimmer:APQ5",
            value=0.025
        )

        apq = st.number_input(
            "MDVP:APQ",
            value=0.03
        )

        dda = st.number_input(
            "Shimmer:DDA",
            value=0.06
        )

        nhr = st.number_input(
            "NHR",
            value=0.02
        )

        hnr = st.number_input(
            "HNR",
            value=20.0
        )

    with col3:

        rpde = st.number_input(
            "RPDE",
            value=0.45
        )

        dfa = st.number_input(
            "DFA",
            value=0.82
        )

        spread1 = st.number_input(
            "spread1",
            value=-4.5
        )

        spread2 = st.number_input(
            "spread2",
            value=0.25
        )

        d2 = st.number_input(
            "D2",
            value=2.35
        )

        ppe = st.number_input(
            "PPE",
            value=0.30
        )

    # --------------------------------------------------
    # Predict Button
    # --------------------------------------------------

    submit = st.form_submit_button(
        "🔍 Analyze",
        use_container_width=True
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submit:

    input_data = pd.DataFrame(
        [[
            fo,
            fhi,
            flo,
            jitter,
            jitter_abs,
            rap,
            ppq,
            ddp,
            shimmer,
            shimmer_db,
            apq3,
            apq5,
            apq,
            dda,
            nhr,
            hnr,
            rpde,
            dfa,
            spread1,
            spread2,
            d2,
            ppe
        ]],
        columns=features
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            input_scaled
        )[0]

        parkinsons_probability = probability[1]
        healthy_probability = probability[0]

    else:

        parkinsons_probability = None
        healthy_probability = None

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.divider()

    st.subheader("📊 Analysis Result")

    if prediction == 1:

        st.error(
            "🔴 Model Classification: Parkinson's Class"
        )

    else:

        st.success(
            "🟢 Model Classification: Healthy Class"
        )

    # --------------------------------------------------
    # Probability
    # --------------------------------------------------

    if parkinsons_probability is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Parkinson's Probability",
                f"{parkinsons_probability:.2%}"
            )

        with col2:

            st.metric(
                "Healthy Probability",
                f"{healthy_probability:.2%}"
            )

        st.subheader("Prediction Probability")

        st.progress(
            float(parkinsons_probability)
        )

    # --------------------------------------------------
    # Input Summary
    # --------------------------------------------------

    with st.expander("View Input Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    st.warning(
        """
        This prediction is generated by a machine-learning model
        trained on a specific research dataset. It should not be
        interpreted as a medical diagnosis or used for clinical
        decision-making.
        """
    )