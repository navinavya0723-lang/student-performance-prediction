"""Streamlit interface for Student Performance Prediction."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "student_performance_model.joblib"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 18px; border-radius: 12px;
        border: 1px solid #e6eaf0;
    }
    .hero {
        padding: 24px; border-radius: 16px; color: white;
        background: linear-gradient(110deg, #173b72, #2878b8);
        margin-bottom: 18px;
    }
    .small-note {color: #64748b; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 Student Performance Prediction</h1>
  <p>Estimate academic performance using machine learning and student study indicators.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Follow the setup instructions in README.md.")
    st.code("python generate_data.py\npython train_model.py\nstreamlit run app.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

st.subheader("Enter student details")
st.write("Adjust the values below, then select **Predict performance**.")

left, right = st.columns(2)
with left:
    study_hours = st.slider("Study hours per day", 0.5, 8.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 45, 100, 80)
    previous_marks = st.slider("Previous examination marks", 0, 100, 65)
with right:
    assignment_completion = st.slider("Assignment completion (%)", 20, 100, 75)
    sleep_hours = st.slider("Sleep hours per day", 4.0, 9.0, 7.0, 0.5)

if st.button("Predict performance", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_completion": assignment_completion,
        "sleep_hours": sleep_hours,
    }])

    predicted_score = float(np.clip(model.predict(input_data)[0], 0, 100))
    rounded_score = round(predicted_score, 1)

    if predicted_score >= 80:
        category = "Excellent"
        advice = "Keep up your consistent study habits and continue revising regularly."
    elif predicted_score >= 65:
        category = "Good"
        advice = "Maintain your routine and focus on improving weaker topics."
    elif predicted_score >= 45:
        category = "Average"
        advice = "Try a consistent study timetable, attend classes regularly, and complete assignments."
    else:
        category = "Needs Improvement"
        advice = "Start with small daily study goals and ask a teacher for support where needed."

    st.divider()
    st.subheader("Prediction results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted score", f"{rounded_score}/100")
    c2.metric("Performance category", category)
    c3.metric("Attendance", f"{attendance}%")

    st.progress(int(round(predicted_score)))
    st.info(f"**Recommendation:** {advice}")

    chart_data = pd.DataFrame({
        "Measure": ["Predicted score", "Previous marks"],
        "Marks": [predicted_score, previous_marks],
    }).set_index("Measure")
    st.bar_chart(chart_data)

    st.caption("This is an educational estimate from a demonstration model, not an official grade or a guarantee of future results.")

with st.expander("About this project"):
    st.write(
        "This project uses a Random Forest Regressor trained on a generated demonstration "
        "dataset. It estimates a score from study hours, attendance, previous marks, "
        "assignment completion, and sleep hours."
    )
    st.warning(
        "The included data is synthetic and created for a mini-project demonstration. "
        "Predictions should not be used to make real academic decisions."
    )

st.markdown("---")
st.markdown(
    "<p class='small-note'>Built with Python, Streamlit, Pandas and Scikit-learn.</p>",
    unsafe_allow_html=True,
)
