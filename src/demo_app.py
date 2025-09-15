import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="AI Dropout Early Warning", page_icon="🎓", layout="centered")

artifacts = joblib.load("model/model.joblib")
MODEL = artifacts["model"]
SCALER = artifacts["scaler"]
FEATURES = artifacts["features"]

st.title("🎓 Student Dropout Early Warning")
st.caption("Calibrated risk score, counselling-aware drivers, and suggested interventions")

col1, col2 = st.columns(2)
with col1:
    gpa = st.slider("GPA (0–10)", 0.0, 10.0, 6.5, 0.1)
    attendance_rate = st.slider("Attendance rate", 0.0, 1.0, 0.75, 0.01)
    lms_logins_14d = st.slider("LMS logins (14 days)", 0, 120, 45, 1)
    late_submit_rate = st.slider("Late submission rate", 0.0, 1.0, 0.25, 0.01)
with col2:
    failed_courses_cum = st.number_input("Failed courses (cumulative)", 0, 20, 1)
    credits_this_term = st.slider("Credits this term", 12, 32, 24, 1)
    dues_days = st.slider("Fee dues (days)", 0, 40, 5, 1)
    help_sessions_14d = st.slider("Help sessions (14 days)", 0, 10, 1, 1)
    counselling_sessions_cum = st.slider("Counselling sessions (cumulative)", 0, 15, 2, 1)
    last_counselling_gap = st.slider("Days since last counselling", 0, 365, 90, 1)
    counselling_notes_score = st.slider("Counselling notes score", 0.0, 1.0, 0.6, 0.01)

x = np.array([[gpa, attendance_rate, lms_logins_14d, late_submit_rate,
               failed_courses_cum, credits_this_term, dues_days, help_sessions_14d,
               counselling_sessions_cum, last_counselling_gap, counselling_notes_score]])
xs = SCALER.transform(x)
p = float(MODEL.predict_proba(xs)[0, 1])

st.markdown(f"### Risk score: {p:.2%}")

# --- FIXED: Handle sklearn version differences ---
coefs = []
for cc in MODEL.calibrated_classifiers_:
    if hasattr(cc, "estimator"):  # new sklearn
        lr = cc.estimator
    elif hasattr(cc, "base_estimator"):  # old sklearn
        lr = cc.base_estimator
    else:
        lr = None
    if lr is not None and hasattr(lr, "coef_"):
        coefs.append(lr.coef_[0])

if coefs:
    coef_mean = np.mean(np.vstack(coefs), axis=0)
    impacts = xs[0] * coef_mean
    top_idx = np.argsort(np.abs(impacts))[::-1][:5]

    st.subheader("Top drivers")
    for i in top_idx:
        direction = "risk ↑" if impacts[i] > 0 else "risk ↓"
        st.write(f"- **{FEATURES[i]}:** {direction}, impact {impacts[i]:+.3f}")

st.subheader("Suggested interventions")
suggestions = []
if attendance_rate < 0.7:
    suggestions.append("Attendance coaching and attendance pledge for the next 2 weeks.")
if late_submit_rate > 0.3:
    suggestions.append("Time-management workshop and deadline reminders.")
if dues_days > 7:
    suggestions.append("Financial aid counselling and fee extension plan.")
if gpa < 6.0 or failed_courses_cum >= 2:
    suggestions.append("Tutor pairing and weekly academic check-ins.")
if help_sessions_14d == 0:
    suggestions.append("Proactive outreach: book first tutoring session.")

# Counselling-aware suggestions
if counselling_sessions_cum == 0:
    suggestions.append("Schedule first counselling session to build rapport.")
elif last_counselling_gap > 120:
    suggestions.append("Follow-up counselling to maintain engagement.")
if counselling_notes_score < 0.4:
    suggestions.append("Escalate to senior counsellor; explore underlying issues.")

if suggestions:
    for s in suggestions[:4]:
        st.write(f"- {s}")
else:
    st.write("- Maintain current plan; optional skill boosters.")
