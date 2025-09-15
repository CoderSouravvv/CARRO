import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

N_STUDENTS = 1200
TERMS = 6

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

rows = []
for sid in range(1, N_STUDENTS + 1):
    gpa = rng.normal(7.2, 1.2)             # 0-10 scale common in India
    engagement = rng.uniform(0.4, 0.9)     # baseline engagement
    financial_risk = rng.binomial(1, 0.25) # dues risk flag
    help_inclination = rng.uniform(0.2, 0.8)
    counselling_base = rng.uniform(0.0, 1.0)

    last_session_day = None
    sessions_cum = 0

    for term in range(1, TERMS + 1):
        # Dynamics
        gpa = clamp(gpa + rng.normal(0, 0.35) - (0.15 if rng.random() < 0.2 else 0), 0, 10)
        attendance = clamp(engagement + rng.normal(0, 0.1), 0, 1)
        logins14 = int(clamp((engagement * 60) + rng.normal(0, 8), 0, 120))
        late_rate = clamp(1 - engagement + rng.normal(0, 0.05), 0, 1)
        failed_cum = int(max(0, rng.poisson(0.2 + (1 - engagement) * 1.5) + (1 if gpa < 5.5 else 0)))
        credits = int(clamp(rng.normal(24, 4), 12, 32))
        dues_days = int(clamp(rng.normal(10 if financial_risk else 2, 6), 0, 40))
        help14 = int(clamp(rng.normal(help_inclination * 3, 1.2), 0, 10))

        # Counselling logs
        # Probability of having a session increases with risk signals
        session_prob = 0.05 + 0.25 * (1 - attendance) + 0.15 * (late_rate) + 0.10 * (dues_days > 7) + 0.10 * (gpa < 6) + 0.05 * counselling_base
        had_session = rng.random() < clamp(session_prob, 0, 0.8)
        if had_session:
            sessions_cum += 1
            last_session_day = rng.integers(0, 60)  # assume a session in last 2 months
        # Days since last counselling
        last_counselling_gap = 365 if last_session_day is None else int(clamp(rng.normal(30 + last_session_day, 20), 0, 365))
        # Notes score trends with student outlook (higher is better)
        counselling_notes_score = round(clamp(rng.normal(0.55 + 0.2 * engagement - 0.2 * (dues_days > 7), 0.2), 0, 1), 2)

        # Dropout probability shaped by risk and mitigations
        risk = 0.05
        risk += 0.22 * (1 - attendance)
        risk += 0.18 * (late_rate)
        risk += 0.20 * (1 if gpa < 6 else 0)
        risk += 0.12 * (failed_cum > 2)
        risk += 0.15 * (dues_days > 7)
        risk += -0.08 * (logins14 > 40)
        risk += -0.06 * (help14 >= 1)
        # Counselling mitigates risk if recent and notes positive; repeated sessions with negative notes can indicate ongoing issues
        risk += -0.07 * (last_counselling_gap <= 60)
        risk += -0.06 * (counselling_notes_score >= 0.6)
        risk += 0.05 * (sessions_cum >= 3 and counselling_notes_score < 0.4)
        risk = clamp(risk + rng.normal(0, 0.04), 0, 0.95)

        dropout_next_term = 1 if rng.random() < risk and term < TERMS else 0

        rows.append({
            "student_id": sid,
            "term_id": term,
            "gpa": round(gpa, 2),
            "attendance_rate": round(attendance, 3),
            "lms_logins_14d": logins14,
            "late_submit_rate": round(late_rate, 3),
            "failed_courses_cum": failed_cum,
            "credits_this_term": credits,
            "dues_days": dues_days,
            "help_sessions_14d": help14,
            "counselling_sessions_cum": sessions_cum,
            "last_counselling_gap": last_counselling_gap,
            "counselling_notes_score": counselling_notes_score,
            "dropout_next_term": dropout_next_term,
            "group_label": rng.choice(["Urban", "Rural"], p=[0.65, 0.35])
        })

data = pd.DataFrame(rows)
Path("data").mkdir(exist_ok=True, parents=True)
data.to_csv("data/student_term.csv", index=False)
print("Wrote data/student_term.csv with", len(data), "rows.")
