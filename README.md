# AI-based student dropout early warning :

Predict next-term dropout risk, explain why, and recommend interventions — with counselling log integration.

## What this demo includes
- Calibrated probabilities and interpretable drivers.
- Counselling log features: sessions, recency gap, and notes score.
- Live FastAPI scoring endpoint.
- Streamlit UI with sliders and adaptive interventions.
- Optional fairness audit across groups.

## How to run
1. pip install -r requirements.txt
2. python src/generate_data.py
3. python src/train.py
4. uvicorn src.service:app --reload --host 0.0.0.0 --port 8000
5. streamlit run src/demo_app.py

## demo flow
- Show `data/student_term.csv` sample rows.
- Run training; point to ROC AUC, Brier, and top drivers.
- Hit `/score` with a risky profile to see risk and top factors.
- Use the UI sliders; watch risk and interventions change.
- (Optional) Run `src/fairness_audit.py` and discuss thresholds.

## Notes
- Sensitive attributes are excluded from training; used only for auditing.
- Human-in-the-loop: scores inform advisors; they decide actions.
