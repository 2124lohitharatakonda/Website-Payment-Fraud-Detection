"""
Deep Learning fraud detection model using scikit-learn MLP.
Covers: architecture definition, training, evaluation, persistence.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
    average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV

from features import FEATURE_NAMES

MODEL_PATH = "models/mlp_fraud.pkl"

# MLP architecture: Input(42) → 256 → 128 → 64 → Output(2)
MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,            # L2 regularisation
    "batch_size": 512,
    "learning_rate": "adaptive",
    "learning_rate_init": 1e-3,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 15,
    "random_state": 42,
    "verbose": False,
}


def build_pipeline() -> Pipeline:
    """Scaler + calibrated MLP wrapped in a sklearn Pipeline."""
    mlp = MLPClassifier(**MLP_PARAMS)
    calibrated = CalibratedClassifierCV(mlp, cv=3, method="sigmoid")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", calibrated),
    ])


def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def cross_validate_model(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        summary[metric] = {
            "mean": float(results[key].mean()),
            "std":  float(results[key].std()),
        }
    return summary


def evaluate(pipe: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":           float((y_pred == y_test).mean()),
        "precision":          float(precision_score(y_test, y_pred)),
        "recall":             float(recall_score(y_test, y_pred)),
        "f1":                 float(f1_score(y_test, y_pred)),
        "auc_roc":            float(roc_auc_score(y_test, y_prob)),
        "avg_precision":      float(average_precision_score(y_test, y_prob)),
    }

    print("\n=== MLP Deep Learning — Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    return metrics


def predict_single(pipe: Pipeline, features: dict) -> dict:
    row = pd.DataFrame([features])[FEATURE_NAMES].values
    prob = pipe.predict_proba(row)[0][1]
    label = "FRAUD" if prob >= 0.5 else "LEGIT"
    risk  = "HIGH" if prob >= 0.75 else ("MEDIUM" if prob >= 0.4 else "LOW")
    return {"probability": round(float(prob), 4), "label": label, "risk": risk}


def predict_batch(pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_NAMES].values
    probs = pipe.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    df = df.copy()
    df["fraud_probability"] = probs.round(4)
    df["prediction"] = preds
    df["risk_level"] = pd.cut(
        probs, bins=[0, 0.4, 0.75, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"], right=True
    )
    return df


def save_model(pipe: Pipeline) -> None:
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from features import batch_extract

    np.random.seed(42)
    n = 8000

    # Synthetic raw records
    records = []
    for i in range(n):
        is_fraud = np.random.choice([0, 1], p=[0.97, 0.03])
        records.append({
            "account_id": np.random.randint(1, 300),
            "amount": float(np.random.exponential(300 if is_fraud else 150)),
            "hour": np.random.choice(range(0, 5)) if is_fraud else np.random.randint(8, 22),
            "ip_country": np.random.choice(["RU","CN","NG"]) if is_fraud else "US",
            "card_country": "US",
            "merchant_category": np.random.choice(["crypto","gambling"]) if is_fraud
                                  else np.random.choice(["retail","food","online"]),
            "card_type": "visa",
            "is_vpn": bool(is_fraud and np.random.rand() > 0.4),
            "is_tor": False,
            "is_new_device": bool(is_fraud and np.random.rand() > 0.5),
            "failed_attempts_1h": np.random.randint(2, 6) if is_fraud else 0,
            "rapid_field_fill": bool(is_fraud),
            "retry_count": np.random.randint(1, 4) if is_fraud else 0,
            "session_duration_sec": float(np.random.randint(5, 20) if is_fraud
                                          else np.random.randint(60, 300)),
            "mouse_velocity_score": float(0.9 + np.random.rand() * 0.1 if is_fraud
                                          else np.random.rand() * 0.5),
            "email": "bot@temp9mail.xyz" if is_fraud else "user@gmail.com",
            "account_verified": not is_fraud,
            "chargeback_history": int(is_fraud),
            "billing_shipping_mismatch": bool(is_fraud and np.random.rand() > 0.5),
            "checkout_time_sec": float(5 if is_fraud else 90),
            "form_revision_count": 0 if is_fraud else np.random.randint(0, 3),
            "referrer_risk": 0.8 if is_fraud else 0.1,
            "device_age_days": np.random.randint(0, 10) if is_fraud else 365,
            "amount_percentile": 0.95 if is_fraud else float(np.random.rand()),
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(records)
    y = df["is_fraud"].values
    X_feat = batch_extract(records, {})

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat.values, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training MLP deep learning model...")
    pipe = train(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test)
    save_model(pipe)

    print("\nCross-validation (5-fold):")
    cv_results = cross_validate_model(X_feat.values, y)
    for metric, vals in cv_results.items():
        print(f"  {metric:<15}: {vals['mean']:.4f} ± {vals['std']:.4f}")
