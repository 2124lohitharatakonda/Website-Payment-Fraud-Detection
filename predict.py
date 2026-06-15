"""
Real-time inference pipeline for web payment fraud detection.
Accepts raw gateway payloads and returns scored results.
"""

import json
import time
from datetime import datetime
import pandas as pd

from features import extract_features, FEATURE_NAMES
from model import load_model, predict_single, predict_batch


BLOCK_THRESHOLD   = 0.75
REVIEW_THRESHOLD  = 0.40


def score_payment(payload: dict, account_history: pd.DataFrame = None) -> dict:
    """
    Score a single web payment for fraud.

    Parameters
    ----------
    payload         : raw payment gateway dict
    account_history : DataFrame of past transactions for this account

    Returns
    -------
    dict with fraud_probability, risk_level, action, latency_ms
    """
    start = time.perf_counter()

    if account_history is None:
        account_history = pd.DataFrame()

    features = extract_features(payload, account_history)

    pipe = load_model()
    result = predict_single(pipe, features)
    prob = result["probability"]

    action = "BLOCK" if prob >= BLOCK_THRESHOLD else (
             "REVIEW" if prob >= REVIEW_THRESHOLD else "APPROVE")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "transaction_id":   payload.get("transaction_id", "N/A"),
        "fraud_probability": prob,
        "risk_level":        result["risk"],
        "action":            action,
        "latency_ms":        latency_ms,
        "timestamp":         datetime.utcnow().isoformat(),
        "top_signals":       _get_top_signals(features, prob),
    }


def _get_top_signals(features: dict, prob: float) -> list[str]:
    """Return human-readable signals that contributed to fraud score."""
    signals = []
    if features.get("is_vpn_proxy"):
        signals.append("VPN/Proxy IP detected")
    if features.get("is_high_risk_country"):
        signals.append("High-risk IP country")
    if features.get("rapid_field_fill"):
        signals.append("Rapid automated form fill")
    if features.get("amount_spike_flag"):
        signals.append("Amount 5× above account average")
    if features.get("failed_attempts_1h", 0) > 1:
        signals.append(f"Failed attempts: {int(features['failed_attempts_1h'])}")
    if features.get("velocity_spike"):
        signals.append("Transaction velocity spike")
    if features.get("billing_shipping_mismatch"):
        signals.append("Billing/shipping address mismatch")
    if features.get("is_new_device"):
        signals.append("Unrecognized device")
    if features.get("chargeback_history", 0) > 0:
        signals.append("Prior chargeback history")
    if features.get("email_domain_risk", 0) > 0.6:
        signals.append("Suspicious email domain")
    return signals[:5]


def score_batch(payloads: list[dict], history_map: dict = None) -> pd.DataFrame:
    """
    Score a batch of payments.

    Parameters
    ----------
    payloads    : list of raw payment dicts
    history_map : {account_id: DataFrame} of past transactions

    Returns
    -------
    DataFrame with one row per payment + score columns
    """
    if history_map is None:
        history_map = {}

    feature_rows = []
    for payload in payloads:
        acc_id = payload.get("account_id")
        history = history_map.get(acc_id, pd.DataFrame())
        feature_rows.append(extract_features(payload, history))

    feat_df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    pipe = load_model()
    scored = predict_batch(pipe, feat_df)

    # Attach original IDs
    scored.insert(0, "transaction_id", [p.get("transaction_id") for p in payloads])
    scored.insert(1, "amount",         [p.get("amount")         for p in payloads])
    scored.insert(2, "merchant",       [p.get("merchant")       for p in payloads])

    scored["action"] = scored["fraud_probability"].apply(
        lambda p: "BLOCK" if p >= BLOCK_THRESHOLD else (
                  "REVIEW" if p >= REVIEW_THRESHOLD else "APPROVE")
    )
    return scored[["transaction_id", "amount", "merchant",
                   "fraud_probability", "risk_level", "action"]]


if __name__ == "__main__":
    # ── Single transaction demo ─────────────────────────────────────────────
    sample_payload = {
        "transaction_id":  "SES-99001",
        "account_id":       42,
        "amount":          4200.0,
        "merchant":        "CryptoFast Exchange",
        "merchant_category": "crypto",
        "card_type":       "visa",
        "card_country":    "US",
        "ip_country":      "RU",
        "is_vpn":          True,
        "is_new_device":   True,
        "failed_attempts_1h": 3,
        "rapid_field_fill": True,
        "retry_count":     2,
        "session_duration_sec": 12.0,
        "mouse_velocity_score": 0.97,
        "email":           "buyer@temp9.xyz",
        "account_verified": False,
        "chargeback_history": 1,
        "billing_shipping_mismatch": True,
        "checkout_time_sec": 7.0,
        "amount_percentile": 0.98,
    }

    try:
        result = score_payment(sample_payload)
        print("=== Single Payment Score ===")
        print(json.dumps(result, indent=2))
    except FileNotFoundError:
        print("Model not found. Run model.py first to train and save the model.")

    # ── Batch demo ─────────────────────────────────────────────────────────
    batch_payloads = [
        {"transaction_id": f"SES-{9000+i}", "account_id": i % 5,
         "amount": float(100 * (i + 1)), "merchant": f"Merchant {i}",
         "merchant_category": "retail", "card_country": "US", "ip_country": "US",
         "is_vpn": False, "card_type": "mastercard"}
        for i in range(5)
    ]

    try:
        batch_results = score_batch(batch_payloads)
        print("\n=== Batch Score Results ===")
        print(batch_results.to_string(index=False))
    except FileNotFoundError:
        print("Model not found. Run model.py first.")
