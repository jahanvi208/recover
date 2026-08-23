"""Feature construction.

One function builds the row for both training and serving, so the two can
never drift apart. Every feature here is derivable from data Razorpay already
has at the moment a payment fails.
"""

from __future__ import annotations

import math

import numpy as np

from . import taxonomy as tx
from .bank_health import BankHealthMonitor

METHODS = ["card", "upi", "netbanking", "wallet"]
NETWORKS = ["visa", "mastercard", "rupay", "amex", "na"]
BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PNB", "YES", "IDFC", "BOB", "CANARA"]
NUDGES = ["none", "email", "sms", "whatsapp"]


def feature_names() -> list[str]:
    names = [
        "log_amount",
        "is_subscription",
        "customer_success_rate",
        "days_to_payday",
        "delay_h",
        "log_delay",
        "delay_days",
        "payday_offset",       # days_to_payday - delay_days
        "retry_local_hour",
        "hour_sin",
        "hour_cos",
        "crosses_midnight",
        "attempt_index",
        "rail_health_now",
        "rail_health_at_retry",
        "rail_degraded",
        "expected_recovery_h",
        "delay_minus_recovery",
    ]
    names += [f"method_{m}" for m in METHODS]
    names += [f"network_{n}" for n in NETWORKS]
    names += [f"bank_{b}" for b in BANKS]
    names += [f"cause_{c}" for c in tx.ROOT_CAUSES]
    names += [f"play_{p}" for p in tx.PLAYS]
    names += [f"nudge_{n}" for n in NUDGES]
    return names


N_FEATURES = len(feature_names())


def _onehot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def build_row(
    pay: dict,
    play: str,
    delay_h: float,
    nudge: str | None,
    attempt_index: int,
    monitor: BankHealthMonitor,
) -> list[float]:
    failed_at = pay["failed_at_h"]
    local_hour = (failed_at + delay_h) % 24
    delay_days = delay_h / 24.0

    # The rail the attempt will actually run on.
    target_method = "upi" if play == tx.SWITCH_RAIL else pay["method"]
    st = monitor.state(pay["bank"], target_method, failed_at)

    row = [
        math.log10(max(pay["amount"], 1.0)),
        1.0 if pay["is_subscription"] else 0.0,
        pay["customer_success_rate"],
        float(pay["days_to_payday"]),
        delay_h,
        math.log1p(delay_h),
        delay_days,
        float(pay["days_to_payday"]) - delay_days,
        local_hour,
        math.sin(2 * math.pi * local_hour / 24),
        math.cos(2 * math.pi * local_hour / 24),
        1.0 if (failed_at % 24) + delay_h >= 24 else 0.0,
        float(attempt_index),
        st.health,
        st.health_at(delay_h),
        1.0 if st.degraded else 0.0,
        st.expected_recovery_h,
        delay_h - st.expected_recovery_h,
    ]
    row += _onehot(pay["method"], METHODS)
    row += _onehot(pay["network"], NETWORKS)
    row += _onehot(pay["bank"], BANKS)
    row += _onehot(pay["cause"], tx.ROOT_CAUSES)
    row += _onehot(play, tx.PLAYS)
    row += _onehot(nudge or "none", NUDGES)
    return row


def build_matrix(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)
