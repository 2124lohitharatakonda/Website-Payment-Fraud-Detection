"""
Feature extraction for web payment transactions.
Covers transaction metadata, behavioral signals, and historical patterns.
"""

import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd


# Known VPN / Tor exit-node ASN prefixes (abbreviated)
SUSPICIOUS_ASNS = {"AS209837", "AS60117", "AS205100", "AS209854"}

# High-risk countries by fraud index
HIGH_RISK_COUNTRIES = {"NG", "RO", "UA", "VN", "PH", "BD", "ID"}

FEATURE_NAMES = [
    "amount_log",
    "hour_sin", "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    "amount_vs_avg_ratio",
    "txn_count_1h",
    "txn_count_24h",
    "failed_attempts_1h",
    "is_new_device",
    "is_vpn_proxy",
    "is_high_risk_country",
    "is_cross_border",
    "card_type_encoded",
    "browser_fingerprint_match",
    "session_duration_sec",
    "mouse_velocity_score",
    "rapid_field_fill",
    "retry_count",
    "merchant_category_encoded",
    "merchant_fraud_rate",
    "amount_percentile",
    "days_since_first_txn",
    "unique_merchants_7d",
    "high_value_flag",
    "velocity_spike",
    "ip_reputation_score",
    "billing_shipping_mismatch",
    "email_domain_risk",
    "card_country_mismatch",
    "page_load_anomaly",
    "checkout_time_sec",
    "form_revision_count",
    "referrer_risk",
    "device_age_days",
    "account_verified",
    "chargeback_history",
    "avg_session_score",
    "proxy_ip_flag",
    "amount_spike_flag",
    "time_since_last_txn_min",
]


def log_amount(amount: float) -> float:
    return float(np.log1p(max(amount, 0)))


def cyclic_encode_hour(hour: int):
    rad = 2 * np.pi * hour / 24
    return float(np.sin(rad)), float(np.cos(rad))


def is_off_hours(hour: int) -> int:
    return int(hour < 6 or hour > 22)


def encode_card_type(card_type: str) -> int:
    mapping = {"visa": 0, "mastercard": 1, "amex": 2, "discover": 3, "other": 4}
    return mapping.get(card_type.lower(), 4)


def encode_merchant_category(category: str) -> int:
    mapping = {
        "retail": 0, "food": 1, "travel": 2, "crypto": 3,
        "gambling": 4, "online": 5, "subscription": 6, "other": 7,
    }
    return mapping.get(category.lower(), 7)


MERCHANT_FRAUD_RATES = {
    "crypto": 0.18, "gambling": 0.14, "travel": 0.07,
    "online": 0.05, "retail": 0.03, "food": 0.01,
    "subscription": 0.02, "other": 0.06,
}


def email_domain_risk(email: str) -> float:
    low_risk = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com"}
    domain = email.split("@")[-1].lower() if "@" in email else ""
    if domain in low_risk:
        return 0.1
    if re.search(r"\d{4,}", domain):  # many numbers in domain → suspicious
        return 0.8
    return 0.4


def browser_fingerprint_match(stored_fp: Optional[str], current_fp: Optional[str]) -> int:
    if not stored_fp or not current_fp:
        return 0
    return int(stored_fp == current_fp)


def velocity_spike(txn_count_1h: int, avg_hourly_count: float) -> int:
    if avg_hourly_count == 0:
        return int(txn_count_1h > 3)
    return int(txn_count_1h / avg_hourly_count > 3.0)


def ip_reputation_score(is_vpn: bool, is_tor: bool, is_high_risk_country: bool) -> float:
    score = 0.0
    if is_vpn:
        score += 0.5
    if is_tor:
        score += 0.4
    if is_high_risk_country:
        score += 0.2
    return min(score, 1.0)


def extract_features(raw: dict, history: pd.DataFrame) -> dict:
    """
    raw      : single transaction dict from the payment gateway
    history  : DataFrame of past transactions for this account
    Returns  : flat feature dict aligned with FEATURE_NAMES
    """
    now = datetime.utcnow()
    hour = raw.get("hour", now.hour)
    amount = float(raw.get("amount", 0))
    sin_h, cos_h = cyclic_encode_hour(hour)

    # History stats
    if not history.empty:
        hist_1h  = history[history["created_at"] >= now - timedelta(hours=1)]
        hist_24h = history[history["created_at"] >= now - timedelta(hours=24)]
        hist_7d  = history[history["created_at"] >= now - timedelta(days=7)]
        avg_amount    = history["amount"].mean()
        days_first    = (now - history["created_at"].min()).days
        unique_merch  = hist_7d["merchant_id"].nunique() if "merchant_id" in hist_7d else 0
        avg_sess_score = history.get("session_score", pd.Series([0.0])).mean()
        time_since_last = (now - history["created_at"].max()).total_seconds() / 60
    else:
        hist_1h = hist_24h = hist_7d = pd.DataFrame()
        avg_amount = amount
        days_first = 0
        unique_merch = 0
        avg_sess_score = 0.0
        time_since_last = 99999.0

    is_vpn = raw.get("is_vpn", False)
    is_tor = raw.get("is_tor", False)
    country = raw.get("ip_country", "US")
    is_high_risk = country in HIGH_RISK_COUNTRIES
    card_country  = raw.get("card_country", "US")

    return {
        "amount_log":                log_amount(amount),
        "hour_sin":                  sin_h,
        "hour_cos":                  cos_h,
        "day_of_week":               now.weekday(),
        "is_weekend":                int(now.weekday() >= 5),
        "is_off_hours":              is_off_hours(hour),
        "amount_vs_avg_ratio":       round(amount / max(avg_amount, 1), 4),
        "txn_count_1h":              len(hist_1h),
        "txn_count_24h":             len(hist_24h),
        "failed_attempts_1h":        int(raw.get("failed_attempts_1h", 0)),
        "is_new_device":             int(raw.get("is_new_device", False)),
        "is_vpn_proxy":              int(is_vpn),
        "is_high_risk_country":      int(is_high_risk),
        "is_cross_border":           int(country != card_country),
        "card_type_encoded":         encode_card_type(raw.get("card_type", "other")),
        "browser_fingerprint_match": browser_fingerprint_match(
                                         raw.get("stored_fp"), raw.get("current_fp")),
        "session_duration_sec":      float(raw.get("session_duration_sec", 120)),
        "mouse_velocity_score":      float(raw.get("mouse_velocity_score", 0.5)),
        "rapid_field_fill":          int(raw.get("rapid_field_fill", False)),
        "retry_count":               int(raw.get("retry_count", 0)),
        "merchant_category_encoded": encode_merchant_category(
                                         raw.get("merchant_category", "other")),
        "merchant_fraud_rate":       MERCHANT_FRAUD_RATES.get(
                                         raw.get("merchant_category", "other"), 0.06),
        "amount_percentile":         float(raw.get("amount_percentile", 0.5)),
        "days_since_first_txn":      days_first,
        "unique_merchants_7d":       unique_merch,
        "high_value_flag":           int(amount > 1000),
        "velocity_spike":            velocity_spike(len(hist_1h),
                                         len(history) / max(days_first, 1) / 24),
        "ip_reputation_score":       ip_reputation_score(is_vpn, is_tor, is_high_risk),
        "billing_shipping_mismatch": int(raw.get("billing_shipping_mismatch", False)),
        "email_domain_risk":         email_domain_risk(raw.get("email", "")),
        "card_country_mismatch":     int(card_country != country),
        "page_load_anomaly":         int(raw.get("page_load_anomaly", False)),
        "checkout_time_sec":         float(raw.get("checkout_time_sec", 60)),
        "form_revision_count":       int(raw.get("form_revision_count", 0)),
        "referrer_risk":             float(raw.get("referrer_risk", 0.1)),
        "device_age_days":           int(raw.get("device_age_days", 365)),
        "account_verified":          int(raw.get("account_verified", True)),
        "chargeback_history":        int(raw.get("chargeback_history", 0)),
        "avg_session_score":         float(avg_sess_score),
        "proxy_ip_flag":             int(is_vpn or is_tor),
        "amount_spike_flag":         int(amount > avg_amount * 5),
        "time_since_last_txn_min":   round(time_since_last, 2),
    }


def batch_extract(records: list[dict], history_map: dict) -> pd.DataFrame:
    rows = [
        extract_features(rec, history_map.get(rec.get("account_id"), pd.DataFrame()))
        for rec in records
    ]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


if __name__ == "__main__":
    sample = {
        "amount": 4500.0, "hour": 3, "ip_country": "RU", "card_country": "US",
        "merchant_category": "crypto", "card_type": "visa",
        "is_vpn": True, "is_tor": False, "is_new_device": True,
        "failed_attempts_1h": 3, "rapid_field_fill": True, "retry_count": 2,
        "session_duration_sec": 14, "mouse_velocity_score": 0.95,
        "email": "user123@temp4mail.xyz", "account_verified": False,
        "chargeback_history": 1, "billing_shipping_mismatch": True,
        "checkout_time_sec": 8, "form_revision_count": 0,
    }
    feats = extract_features(sample, pd.DataFrame())
    for k, v in feats.items():
        print(f"  {k:<35}: {v}")
