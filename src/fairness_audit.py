import pandas as pd
import joblib
from utils import load_dataset, transform, FEATURES

ART = joblib.load("model/model.joblib")
MODEL = ART["model"]; SCALER = ART["scaler"]

X, y, meta, df = load_dataset()
if "group_label" not in meta:
    print("No group_label found. Skipping audit.")
    raise SystemExit()

# Use the last term as a pseudo-future holdout
last_term = df["term_id"].max()
mask = df["term_id"] == last_term
Xte = df.loc[mask, FEATURES]
yte = df.loc[mask, "dropout_next_term"].astype(int)
gte = df.loc[mask, "group_label"]

probs = MODEL.predict_proba(transform(SCALER, Xte))[:, 1]
d = pd.DataFrame({"y": yte.values, "p": probs, "g": gte.values})
thr = 0.30
d["pred"] = (d["p"] >= thr).astype(int)

def metrics(grp):
    tp = ((grp.y==1)&(grp.pred==1)).sum()
    fn = ((grp.y==1)&(grp.pred==0)).sum()
    fp = ((grp.y==0)&(grp.pred==1)).sum()
    tn = ((grp.y==0)&(grp.pred==0)).sum()
    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    return pd.Series({"count": len(grp), "TPR": round(tpr,3), "FPR": round(fpr,3), "Precision": round(prec,3)})

print("Fairness metrics at threshold =", thr)
print(d.groupby("g").apply(metrics))
