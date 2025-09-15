from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Dropout Risk Scoring API", version="1.1")

ARTIFACTS = joblib.load("model/model.joblib")
MODEL = ARTIFACTS["model"]
SCALER = ARTIFACTS["scaler"]
FEATURES = ARTIFACTS["features"]

class StudentPayload(BaseModel):
    gpa: float = Field(..., ge=0, le=10)
    attendance_rate: float = Field(..., ge=0, le=1)
    lms_logins_14d: int = Field(..., ge=0)
    late_submit_rate: float = Field(..., ge=0, le=1)
    failed_courses_cum: int = Field(..., ge=0)
    credits_this_term: int = Field(..., ge=0)
    dues_days: int = Field(..., ge=0)
    help_sessions_14d: int = Field(..., ge=0)
    counselling_sessions_cum: int = Field(..., ge=0)
    last_counselling_gap: int = Field(..., ge=0)
    counselling_notes_score: float = Field(..., ge=0, le=1)

@app.get("/")
def root():
    return {"status": "ok", "features": FEATURES}

@app.post("/score")
def score(payload: StudentPayload):
    x = np.array([[getattr(payload, f) for f in FEATURES]])
    xs = SCALER.transform(x)
    p = float(MODEL.predict_proba(xs)[0, 1])

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

    reasons = []
    if coefs:
        coef_mean = np.mean(np.vstack(coefs), axis=0)
        impacts = xs[0] * coef_mean
        contrib_sorted = sorted(list(zip(FEATURES, impacts)), key=lambda t: abs(t[1]), reverse=True)[:5]
        reasons = [
            {"feature": f, "direction": "risk↑" if v > 0 else "risk↓", "impact": round(float(v), 3)}
            for f, v in contrib_sorted
        ]

    return {"risk_score": round(p, 4), "top_factors": reasons}
