import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "gpa", "attendance_rate", "lms_logins_14d", "late_submit_rate",
    "failed_courses_cum", "credits_this_term", "dues_days", "help_sessions_14d",
    "counselling_sessions_cum", "last_counselling_gap", "counselling_notes_score"
]

def load_dataset(path="data/student_term.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["dropout_next_term"])
    X = df[FEATURES].fillna(0)
    y = df["dropout_next_term"].astype(int)
    meta = {}
    if "group_label" in df.columns:
        meta["group_label"] = df["group_label"]
    return X, y, meta, df

def fit_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler

def transform(scaler, X):
    return scaler.transform(X.values)

def save_feature_meta(scaler, path="model/feature_meta.json"):
    means = scaler.mean_.tolist()
    scales = scaler.scale_.tolist()
    obj = {"features": FEATURES, "scaler_mean": means, "scaler_scale": scales}
    with open(path, "w") as f:
        json.dump(obj, f)

def load_feature_meta(path="model/feature_meta.json"):
    with open(path, "r") as f:
        return json.load(f)
