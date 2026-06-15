-- ============================================================
-- PayGuard — Web Payment Fraud Detection Schema & Queries
-- ============================================================

-- ------------------------------------------------------------
-- Schema
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    id               VARCHAR(36)  PRIMARY KEY,
    account_id       INT          NOT NULL,
    ip_address       INET,
    ip_country       CHAR(2),
    is_vpn           BOOLEAN      DEFAULT FALSE,
    is_tor           BOOLEAN      DEFAULT FALSE,
    device_id        VARCHAR(128),
    browser_fp       VARCHAR(256),
    session_start    TIMESTAMP    DEFAULT NOW(),
    session_end      TIMESTAMP,
    duration_sec     INT
);

CREATE TABLE IF NOT EXISTS payments (
    id                   VARCHAR(36)   PRIMARY KEY,
    session_id           VARCHAR(36)   REFERENCES sessions(id),
    account_id           INT           NOT NULL,
    amount               NUMERIC(15,2) NOT NULL,
    currency             CHAR(3)       DEFAULT 'USD',
    merchant_id          VARCHAR(64),
    merchant_category    VARCHAR(64),
    card_type            VARCHAR(20),
    card_country         CHAR(2),
    billing_country      CHAR(2),
    shipping_country     CHAR(2),
    fraud_probability    NUMERIC(5,4),
    risk_level           VARCHAR(10),
    action               VARCHAR(10),   -- APPROVE | REVIEW | BLOCK
    signal_flags         TEXT[],
    model_version        VARCHAR(20),
    inference_ms         NUMERIC(6,2),
    created_at           TIMESTAMP     DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS behavioral_signals (
    id                   SERIAL        PRIMARY KEY,
    payment_id           VARCHAR(36)   REFERENCES payments(id),
    mouse_velocity       NUMERIC(5,3),
    keystroke_rhythm     NUMERIC(5,3),
    rapid_fill           BOOLEAN       DEFAULT FALSE,
    form_revisions       INT           DEFAULT 0,
    checkout_duration_sec NUMERIC(8,2),
    retry_count          INT           DEFAULT 0,
    failed_attempts_1h   INT           DEFAULT 0,
    recorded_at          TIMESTAMP     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pay_account    ON payments(account_id);
CREATE INDEX IF NOT EXISTS idx_pay_action     ON payments(action);
CREATE INDEX IF NOT EXISTS idx_pay_prob       ON payments(fraud_probability DESC);
CREATE INDEX IF NOT EXISTS idx_pay_created    ON payments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pay_country    ON payments(ip_country);


-- ------------------------------------------------------------
-- Query 1: Flagged high-risk transactions last 24h
-- ------------------------------------------------------------

SELECT
    p.id               AS payment_id,
    p.account_id,
    p.amount,
    p.merchant_category,
    s.ip_address,
    s.ip_country,
    s.is_vpn,
    p.fraud_probability,
    p.risk_level,
    p.action,
    p.signal_flags,
    p.inference_ms,
    p.created_at
FROM payments p
LEFT JOIN sessions s ON s.id = p.session_id
WHERE p.created_at >= NOW() - INTERVAL '24 hours'
  AND p.fraud_probability >= 0.75
ORDER BY p.fraud_probability DESC
LIMIT 200;


-- ------------------------------------------------------------
-- Query 2: Detection stats by hour (last 7 days)
-- ------------------------------------------------------------

SELECT
    DATE_TRUNC('hour', created_at)                        AS hour_bucket,
    COUNT(*)                                              AS total_payments,
    SUM(CASE WHEN action = 'BLOCK'  THEN 1 ELSE 0 END)   AS blocked,
    SUM(CASE WHEN action = 'REVIEW' THEN 1 ELSE 0 END)   AS reviewed,
    SUM(CASE WHEN action = 'APPROVE'THEN 1 ELSE 0 END)   AS approved,
    ROUND(AVG(fraud_probability) * 100, 2)                AS avg_risk_pct,
    ROUND(AVG(inference_ms), 2)                           AS avg_latency_ms
FROM payments
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour_bucket DESC;


-- ------------------------------------------------------------
-- Query 3: VPN/Proxy transactions and their outcomes
-- ------------------------------------------------------------

SELECT
    s.ip_country,
    s.is_vpn,
    COUNT(p.id)                                            AS total,
    SUM(CASE WHEN p.action = 'BLOCK' THEN 1 ELSE 0 END)   AS blocked,
    ROUND(AVG(p.fraud_probability), 4)                     AS avg_fraud_prob,
    SUM(p.amount)                                          AS total_attempted_value
FROM payments p
JOIN sessions s ON s.id = p.session_id
WHERE p.created_at >= NOW() - INTERVAL '30 days'
GROUP BY s.ip_country, s.is_vpn
ORDER BY avg_fraud_prob DESC
LIMIT 20;


-- ------------------------------------------------------------
-- Query 4: Behavioral signal correlation with fraud
-- ------------------------------------------------------------

SELECT
    CASE
        WHEN b.rapid_fill AND b.failed_attempts_1h > 1 THEN 'Bot-like'
        WHEN b.rapid_fill THEN 'Rapid Fill'
        WHEN b.failed_attempts_1h > 1 THEN 'Failed Retries'
        ELSE 'Normal'
    END                                                    AS behavior_class,
    COUNT(*)                                               AS count,
    ROUND(AVG(p.fraud_probability), 4)                     AS avg_fraud_prob,
    ROUND(AVG(b.checkout_duration_sec), 1)                 AS avg_checkout_sec,
    SUM(CASE WHEN p.action = 'BLOCK' THEN 1 ELSE 0 END)   AS blocked_count
FROM payments p
JOIN behavioral_signals b ON b.payment_id = p.id
WHERE p.created_at >= NOW() - INTERVAL '14 days'
GROUP BY behavior_class
ORDER BY avg_fraud_prob DESC;


-- ------------------------------------------------------------
-- Query 5: Merchant category risk ranking
-- ------------------------------------------------------------

SELECT
    merchant_category,
    COUNT(*)                                               AS total_transactions,
    ROUND(AVG(fraud_probability), 4)                       AS avg_fraud_prob,
    SUM(CASE WHEN action = 'BLOCK' THEN 1 ELSE 0 END)     AS blocked,
    ROUND(
        SUM(CASE WHEN action = 'BLOCK' THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) * 100, 2
    )                                                      AS block_rate_pct
FROM payments
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY merchant_category
ORDER BY avg_fraud_prob DESC;
