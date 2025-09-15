import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

from utils import load_dataset, fit_scaler, transform, save_feature_meta, FEATURES

Path("model").mkdir(exist_ok=True)

# Load data
X, y, meta, df = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# Scale features
scaler = fit_scaler(X_train)
Xtr = transform(scaler, X_train)
Xte = transform(scaler, X_test)

# Train calibrated logistic regression
base = LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear")
clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
clf.fit(Xtr, y_train)

# Evaluate
probs = clf.predict_proba(Xte)[:, 1]
auc = roc_auc_score(y_test, probs)
brier = brier_score_loss(y_test, probs)

print(f"ROC AUC: {auc:.3f}")
print(f"Brier score: {brier:.3f}")

# Save artifacts
joblib.dump({"model": clf, "scaler": scaler, "features": FEATURES}, "model/model.joblib")
save_feature_meta(scaler)

# --- FIXED: Handle sklearn version differences for coefficient extraction ---
coefs = []
for cc in clf.calibrated_classifiers_:
    if hasattr(cc, "estimator"):  # scikit-learn >= 1.6
        lr = cc.estimator
    elif hasattr(cc, "base_estimator"):  # older versions
        lr = cc.base_estimator
    else:
        raise AttributeError("Cannot find underlying estimator in calibrated classifier.")
    if hasattr(lr, "coef_"):
        coefs.append(lr.coef_[0])

if coefs:
    coef_mean = np.mean(np.vstack(coefs), axis=0)
    global_importance = sorted(zip(FEATURES, coef_mean), key=lambda x: abs(x[1]), reverse=True)

    print("Top global drivers:")
    for f, w in global_importance[:7]:
        print(f"  {f}: weight {w:+.3f}")

    with open("model/metrics.json", "w") as f:
        json.dump({"roc_auc": float(auc), "brier": float(brier), "top_features": global_importance}, f, indent=2)
else:
    print("No coefficients found for the model.")
