import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Model
# =========================

model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Prediction Dashboard")

tab1, tab2 = st.tabs([
    "Prediksi Placement",
    "Insight Clustering"
])

# ===================================================
# TAB 1 - PREDICTION
# ===================================================

with tab1:

    st.header("Prediksi Status Placement")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider(
            "Study Hours",
            0.0, 12.0, 6.0
        )

        attendance = st.slider(
            "Attendance (%)",
            0, 100, 75
        )

        sleep_hours = st.slider(
            "Sleep Hours",
            3.0, 10.0, 7.0
        )

        internet_usage = st.slider(
            "Internet Usage",
            0.0, 12.0, 5.0
        )

        assignments_completed = st.slider(
            "Assignments Completed",
            0, 20, 10
        )

    with col2:

        previous_score = st.slider(
            "Previous Score",
            0, 100, 70
        )

        productive_score = st.slider(
            "Productive Score",
            0.0, 10.0, 5.0
        )

        academic_composite = st.slider(
            "Academic Composite",
            0.0, 1.0, 0.7
        )

        digital_balance = st.slider(
            "Digital Balance",
            -10.0, 10.0, 0.0
        )

        sleep_quality = st.slider(
            "Sleep Quality",
            0.0, 1.0, 0.6
        )

    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "internet_usage": [internet_usage],
        "assignments_completed": [assignments_completed],
        "previous_score": [previous_score],
        "productive_score": [productive_score],
        "academic_composite": [academic_composite],
        "digital_balance": [digital_balance],
        "sleep_quality": [sleep_quality]
    })

    if st.button("Prediksi"):

        scaled = scaler.transform(input_data)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        st.subheader("Hasil Prediksi")

        if prediction == 1:
            st.success("✅ PLACED")
        else:
            st.error("❌ NOT PLACED")

        st.metric(
            "Probabilitas Placement",
            f"{probability*100:.2f}%"
        )

        st.progress(float(probability))

# ===================================================
# TAB 2 - CLUSTERING
# ===================================================

with tab2:

    st.header("Insight Clustering Mahasiswa")

    cluster_df = pd.DataFrame({
        "Cluster": [
            "Mahasiswa Kurang Aktif",
            "Mahasiswa Aktif & Produktif"
        ],
        "Study Hours": [3.54, 8.34],
        "Attendance": [69.27, 70.47],
        "Productive Score": [4.27, 7.64],
        "Digital Balance": [-3.23, 2.96],
        "Placement Rate (%)": [68.12, 98.37]
    })

    st.dataframe(
        cluster_df,
        use_container_width=True
    )

    st.subheader("Karakteristik Cluster")

    c1, c2 = st.columns(2)

    with c1:
        st.info("""
        ### Cluster 0
        **Mahasiswa Kurang Aktif**

        - Study Hours rendah
        - Productive Score rendah
        - Digital Balance negatif
        - Placement Rate 68.12%
        """)

    with c2:
        st.success("""
        ### Cluster 1
        **Mahasiswa Aktif dan Produktif**

        - Study Hours tinggi
        - Productive Score tinggi
        - Digital Balance positif
        - Placement Rate 98.37%
        """)

    st.subheader("Feature Importance Model")

    importance = pd.DataFrame({
        "Feature": [
            "productive_score",
            "study_hours",
            "academic_composite",
            "digital_balance",
            "assignments_completed",
            "previous_score",
            "attendance",
            "internet_usage",
            "sleep_hours",
            "sleep_quality"
        ],
        "Importance": [
            0.280595,
            0.149450,
            0.136202,
            0.118319,
            0.083188,
            0.071234,
            0.066187,
            0.040913,
            0.038618,
            0.015293
        ]
    })

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.markdown("""
    ### Kesimpulan

    Faktor yang paling berpengaruh terhadap peluang placement adalah:

    1. Productive Score
    2. Study Hours
    3. Academic Composite
    4. Digital Balance

    Mahasiswa pada Cluster Aktif & Produktif memiliki peluang placement
    hampir 98%, jauh lebih tinggi dibandingkan cluster lainnya.
    """)
