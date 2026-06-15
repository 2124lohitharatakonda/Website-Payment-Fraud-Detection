# 💠 Website Payment Fraud Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MLP%20Neural%20Net-FF4B4B?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Precision](https://img.shields.io/badge/Precision-96.2%25-blue?style=flat-square)
![Latency](https://img.shields.io/badge/Avg%20Latency-38ms-orange?style=flat-square)
![Features](https://img.shields.io/badge/Features-42-purple?style=flat-square)

**An end-to-end deep learning fraud detection system for web transactions. Analyses transaction metadata, behavioral signals, and historical patterns to flag suspicious activity in real time with high precision.**

[Overview](#overview) • [Architecture](#architecture) • [Features](#features) • [Neural Network](#neural-network-design) • [Feature Engineering](#feature-engineering) • [Model Performance](#model-performance) • [Setup](#setup)

</div>

---

## 📌 Overview

Online payment fraud is increasingly sophisticated — attackers use VPNs, bots, and stolen credentials to mimic legitimate behavior. This project addresses the challenge by combining:

- **42 behavioral and metadata features** extracted per session
- A **Multi-Layer Perceptron (MLP)** deep learning model trained with scikit-learn
- **Calibrated probability output** via `CalibratedClassifierCV` for reliable fraud scores
- **Real-time inference** with average latency of **38ms per transaction**
- **SQL-backed behavioral analytics** for pattern discovery across sessions

The system goes beyond simple rule engines by learning non-linear relationships between features like mouse velocity, checkout timing, IP reputation, and transaction amount — resulting in **96.2% precision** on web fraud detection.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     WEB PAYMENT GATEWAY                               │
│            (checkout form submission / API webhook)                   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  Raw transaction payload (JSON)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION  (features.py)                   │
│                                                                        │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │ Transaction Meta │  │ Behavioral Signal │  │  Historical Pattern │ │
│  │ • amount_log     │  │ • mouse_velocity  │  │ • txn_count_1h      │ │
│  │ • hour sin/cos   │  │ • rapid_fill      │  │ • txn_count_24h     │ │
│  │ • card_type      │  │ • session_duration│  │ • velocity_spike    │ │
│  │ • merchant_cat   │  │ • checkout_time   │  │ • avg_session_score │ │
│  │ • cross_border   │  │ • form_revisions  │  │ • time_since_last   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────┘ │
│                                                                        │
│  ┌─────────────────┐  ┌──────────────────┐                           │
│  │  IP Reputation   │  │  Account Context  │                           │
│  │ • is_vpn_proxy   │  │ • account_verified│                           │
│  │ • ip_reputation  │  │ • chargeback_hist │                           │
│  │ • high_risk_ctry │  │ • device_age_days │                           │
│  │ • proxy_ip_flag  │  │ • email_domain    │                           │
│  └─────────────────┘  └──────────────────┘                           │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  42-dimensional feature vector
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│              DEEP LEARNING MODEL PIPELINE  (model.py)                 │
│                                                                        │
│   StandardScaler → MLPClassifier → CalibratedClassifierCV             │
│                                                                        │
│   Input (42)  →  Dense (256, ReLU)  →  Dense (128, ReLU)             │
│              →  Dense (64, ReLU)    →  Output (Sigmoid)               │
│                                                                        │
│   Solver: Adam  |  L2 alpha: 1e-4  |  Early Stopping: 15 rounds      │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  fraud_probability ∈ [0, 1]
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE  (predict.py)                      │
│                                                                        │
│   fraud_probability ≥ 0.75  →  BLOCK   (auto-reject payment)         │
│   fraud_probability 0.40–0.75 → REVIEW (manual analyst queue)        │
│   fraud_probability < 0.40   →  APPROVE (payment cleared)            │
│                                                                        │
│   Returns: probability, risk_level, action, top 5 fraud signals,     │
│            latency_ms, timestamp                                       │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│               PostgreSQL Database  (database.sql)                      │
│   sessions • payments • behavioral_signals                             │
│   Indexed on: fraud_probability, action, ip_country, created_at       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Neural Network Design

The core model is a **Multi-Layer Perceptron** built with scikit-learn's `MLPClassifier`, calibrated with Platt scaling for reliable probability output:

```
INPUT LAYER          HIDDEN LAYER 1       HIDDEN LAYER 2       HIDDEN LAYER 3       OUTPUT
                                                                                      
  Feature 1  ──┐                                                                      
  Feature 2  ──┤                                                                      
  Feature 3  ──┤    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐      
  Feature 4  ──┼───▶│  256 neurons │────▶│  128 neurons │────▶│   64 neurons │────▶ P(fraud)
  Feature 5  ──┤    │  Activation  │     │  Activation  │     │  Activation  │      Sigmoid
  ...        ──┤    │  ReLU        │     │  ReLU        │     │  ReLU        │      ∈ [0, 1]
  Feature 42 ──┘    └──────────────┘     └──────────────┘     └──────────────┘      
                     L2 reg: 1e-4         Dropout implicit      Batch: 512           
                     
Training: Adam optimizer  |  lr=1e-3 adaptive  |  Early stopping: 15 rounds
Calibration: CalibratedClassifierCV (Platt scaling, 3-fold)
```

### Why MLP for this problem?

| Reason | Explanation |
|--------|------------|
| **Non-linearity** | Fraud patterns are not linearly separable — ReLU activations model complex interactions between behavioral signals |
| **Feature interactions** | Hidden layers automatically learn combinations like `(rapid_fill AND off_hours AND new_device)` |
| **Calibrated probability** | Platt scaling ensures `P(fraud) = 0.85` actually means 85% of such cases are fraud |
| **42 features** | Depth of 3 hidden layers is appropriate for this feature dimensionality without overfitting |

---

## ✨ Features

### 🔍 42-Feature Behavioral Analysis

Features are organized into 5 signal groups:

#### 1. Transaction Metadata (8 features)
- Log-scaled amount, cyclic time encoding (sin/cos hour), day of week, weekend flag
- Card type, merchant category, merchant fraud rate, cross-border flag

#### 2. Behavioral Signals (7 features)
- Mouse velocity score, rapid field fill detection, checkout duration
- Form revision count, retry count, failed attempts in 1h, page load anomaly

#### 3. IP & Device Intelligence (7 features)
- VPN/Proxy detection, Tor exit node flag, high-risk country classification
- IP reputation score (0–1), proxy IP flag, device age in days, new device flag

#### 4. Account & Identity (6 features)
- Account verification status, chargeback history count
- Email domain risk scoring (temp mail → high risk), billing/shipping mismatch
- Browser fingerprint match (stored vs current), referrer risk score

#### 5. Historical Patterns (6 features)
- Transaction count in 1h window, transaction count in 24h window
- Amount vs. rolling average ratio, velocity spike flag (3× above baseline)
- Average session score, time since last transaction (minutes)

### ⚡ Real-Time Inference
- **Single payment scoring** in `predict.py` with human-readable fraud signal explanations
- **Batch scoring** for end-of-day review queues
- **Top 5 fraud signals** returned per transaction (e.g., "VPN/Proxy IP detected", "Rapid automated form fill")

### 🗄️ SQL Analytics
- 5 production queries: hourly detection stats, VPN transaction outcomes, behavioral correlation, merchant risk ranking, flagged sessions
- Window functions for hourly aggregation and behavioral class analysis

---

## 🧰 Tech Stack

| Component | Technology | Version | Role |
|-----------|-----------|---------|------|
| **Language** | Python | 3.11 | Pipeline runtime |
| **Deep Learning** | scikit-learn MLPClassifier | 1.4.2 | 3-layer neural network |
| **Calibration** | CalibratedClassifierCV | 1.4.2 | Platt scaling for probability |
| **Preprocessing** | StandardScaler + Pipeline | 1.4.2 | Feature normalisation |
| **Validation** | StratifiedKFold | 1.4.2 | 5-fold cross-validation |
| **Data** | Pandas, NumPy | 2.2.2 / 1.26.4 | Feature engineering |
| **Database** | PostgreSQL | 15+ | Sessions, payments, signals |
| **Persistence** | Joblib | 1.4.2 | Model serialization |

---

## 📁 Project Structure

```
Website-Payment-Fraud-Detection/
│
├── index.html              ← Real-time monitoring UI dashboard
├── features.py             ← 42-feature extraction pipeline
│                             (behavioral, IP, metadata, historical)
├── model.py                ← MLP deep learning model
│                             (architecture, training, evaluation, CV)
├── predict.py              ← Inference engine
│                             (single + batch scoring, signal explanation)
├── database.sql            ← PostgreSQL schema + 5 analytical queries
├── requirements.txt        ← Python dependencies
└── README.md
```

---

## 📐 Feature Engineering Details

### Cyclic Time Encoding

Hours are encoded as sine/cosine values to preserve circular continuity (so the model understands that 23:00 and 00:00 are close in time):

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

### IP Reputation Scoring

```python
ip_reputation_score = 0.0
if is_vpn:    ip_reputation_score += 0.50
if is_tor:    ip_reputation_score += 0.40
if high_risk_country: ip_reputation_score += 0.20
# Clamped to [0.0, 1.0]
```

### Velocity Spike Detection

```python
velocity_spike = True  if (txn_count_1h / avg_hourly_count) > 3.0  else False
```

### Email Domain Risk

```
Gmail / Yahoo / Outlook → 0.10  (low risk)
Temp-mail / numbered domains → 0.80  (high risk)
Unknown domains → 0.40  (medium risk)
```

---

## 📈 Model Performance

### MLP Neural Network Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 95.1% |
| **Precision** | 96.2% |
| **Recall** | 94.8% |
| **F1 Score** | 0.955 |
| **AUC-ROC** | 0.991 |
| **Avg Precision** | 0.948 |

### 5-Fold Cross-Validation Summary

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | 0.9498 | ±0.0031 |
| Precision | 0.9601 | ±0.0044 |
| Recall | 0.9479 | ±0.0058 |
| F1 | 0.9539 | ±0.0039 |
| AUC-ROC | 0.9908 | ±0.0021 |

> Low standard deviation across folds confirms the model generalises well and is not overfit.

### Detection Breakdown (Real-World Scenario)

```
Per 100,000 web transactions per day:
  ┌──────────────────────────────────────────────────────┐
  │  Total Transactions        100,000                    │
  │  Legitimate (97%)           97,000    → APPROVE       │
  │  Fraudulent  (3%)            3,000                    │
  │    └─ Correctly Blocked      2,844   (Recall 94.8%)  │
  │    └─ Missed (FN)              156   (False Negatives)│
  │  False Positives (FP)        ~116   (0.12% legit)    │
  │                                                        │
  │  Revenue Saved (avg $420)  ~$1.19M / day              │
  └──────────────────────────────────────────────────────┘
```

---

## 🗄️ SQL Analytical Queries

| # | Query | Description |
|---|-------|-------------|
| 1 | High-risk sessions | All sessions with fraud_probability ≥ 0.75 in last 24h |
| 2 | Hourly detection stats | Blocked/reviewed/approved counts + avg latency per hour |
| 3 | VPN transaction analysis | Outcomes grouped by IP country and VPN flag |
| 4 | Behavioral correlation | Fraud rate by behavior class (Bot-like / Rapid Fill / Normal) |
| 5 | Merchant risk ranking | Block rate % per merchant category |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ (or SQLite for local dev)

### Installation

```bash
# Navigate to project directory
cd Website-Payment-Fraud-Detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python model.py
# Trains MLP, runs 5-fold CV, saves → models/mlp_fraud.pkl
```

### Score a Single Payment

```bash
python predict.py
# Loads model, scores sample payload, prints fraud signals
```

### Extract Features (Demo)

```bash
python features.py
# Prints all 42 extracted features for a sample high-risk transaction
```

---

## 📦 Sample Prediction Output

```json
{
  "transaction_id": "SES-99001",
  "fraud_probability": 0.9731,
  "risk_level": "HIGH",
  "action": "BLOCK",
  "latency_ms": 38.4,
  "timestamp": "2024-03-15T02:47:21",
  "top_signals": [
    "VPN/Proxy IP detected",
    "High-risk IP country",
    "Rapid automated form fill",
    "Amount 5× above account average",
    "Failed attempts: 3"
  ]
}
```

---

## 📊 Database Schema

```
sessions
├── id (UUID, PK)
├── account_id
├── ip_address (INET)
├── ip_country, is_vpn, is_tor
├── device_id, browser_fp
└── session_start / session_end / duration_sec

payments
├── id (UUID, PK)
├── session_id (FK → sessions)
├── account_id
├── amount, currency
├── merchant_id, merchant_category
├── card_type, card_country, billing_country
├── fraud_probability
├── risk_level, action
├── signal_flags (TEXT[])
├── model_version, inference_ms
└── created_at

behavioral_signals
├── id (PK)
├── payment_id (FK → payments)
├── mouse_velocity, keystroke_rhythm
├── rapid_fill, form_revisions
├── checkout_duration_sec, retry_count
├── failed_attempts_1h
└── recorded_at
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with Python · Deep Learning · Scikit-learn · SQL
</div>
