"""Failure taxonomy.

Maps raw gateway failure reasons onto root causes. The root cause is what
determines *how* a payment can be recovered -- an expired card and a bank
outage are both "payment failed", but retrying one is free money and
retrying the other is a wasted attempt against a network retry cap.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Root causes
# ---------------------------------------------------------------------------

BANK_DOWNTIME = "bank_downtime"
SOFT_FUNDS = "soft_funds"
AUTH_FRICTION = "auth_friction"
INSTRUMENT_DEAD = "instrument_dead"
RISK_BLOCK = "risk_block"
TECHNICAL = "technical"
LIMIT_EXCEEDED = "limit_exceeded"
USER_ABANDON = "user_abandon"

ROOT_CAUSES = [
    BANK_DOWNTIME,
    SOFT_FUNDS,
    AUTH_FRICTION,
    INSTRUMENT_DEAD,
    RISK_BLOCK,
    TECHNICAL,
    LIMIT_EXCEEDED,
    USER_ABANDON,
]

# ---------------------------------------------------------------------------
# Recovery plays
# ---------------------------------------------------------------------------

RETRY = "retry"                       # same instrument, later
SWITCH_RAIL = "switch_rail"           # card -> UPI (or netbanking -> UPI)
REQUEST_INSTRUMENT = "request_instrument"  # ask the customer for a new card/VPA
SUPPRESS = "suppress"                 # do not touch: risk or hard decline

PLAYS = [RETRY, SWITCH_RAIL, REQUEST_INSTRUMENT, SUPPRESS]

# ---------------------------------------------------------------------------
# Gateway reason -> root cause
#
# Reason strings follow Razorpay's public `error_reason` / `error_description`
# vocabulary closely enough to be recognisable to anyone who has read the
# Payments API docs.
# ---------------------------------------------------------------------------

REASON_TO_CAUSE = {
    "issuer_down": BANK_DOWNTIME,
    "gateway_technical_error": BANK_DOWNTIME,
    "upi_psp_down": BANK_DOWNTIME,
    "npci_unavailable": BANK_DOWNTIME,
    "insufficient_funds": SOFT_FUNDS,
    "account_balance_low": SOFT_FUNDS,
    "payment_authentication_failed": AUTH_FRICTION,
    "incorrect_otp": AUTH_FRICTION,
    "otp_attempts_exceeded": AUTH_FRICTION,
    "three_ds_failed": AUTH_FRICTION,
    "card_expired": INSTRUMENT_DEAD,
    "card_blocked": INSTRUMENT_DEAD,
    "invalid_vpa": INSTRUMENT_DEAD,
    "account_closed": INSTRUMENT_DEAD,
    "international_transaction_not_allowed": INSTRUMENT_DEAD,
    "payment_declined_by_risk": RISK_BLOCK,
    "suspected_fraud": RISK_BLOCK,
    "server_error": TECHNICAL,
    "network_error": TECHNICAL,
    "payment_timed_out": TECHNICAL,
    "per_transaction_limit_exceeded": LIMIT_EXCEEDED,
    "daily_limit_exceeded": LIMIT_EXCEEDED,
    "upi_collect_expired": USER_ABANDON,
    "customer_cancelled": USER_ABANDON,
    "mandate_paused": USER_ABANDON,
}

REASONS = sorted(REASON_TO_CAUSE)


def classify(reason: str) -> str:
    """Return the root cause for a gateway failure reason."""
    return REASON_TO_CAUSE.get(reason, TECHNICAL)


# ---------------------------------------------------------------------------
# Which plays are even legal for a given cause
#
# This is a hard constraint layer, not a preference. Retrying a risk decline
# is a compliance problem, and retrying a dead instrument burns an attempt
# against Visa's 15-attempts-per-30-days cap for no possible upside.
# ---------------------------------------------------------------------------

ALLOWED_PLAYS = {
    BANK_DOWNTIME: [RETRY, SWITCH_RAIL],
    SOFT_FUNDS: [RETRY, REQUEST_INSTRUMENT],
    AUTH_FRICTION: [RETRY, SWITCH_RAIL, REQUEST_INSTRUMENT],
    INSTRUMENT_DEAD: [REQUEST_INSTRUMENT, SWITCH_RAIL],
    RISK_BLOCK: [SUPPRESS],
    TECHNICAL: [RETRY, SWITCH_RAIL],
    LIMIT_EXCEEDED: [RETRY, SWITCH_RAIL],
    USER_ABANDON: [RETRY, SWITCH_RAIL, REQUEST_INSTRUMENT],
}


def allowed_plays(cause: str) -> list[str]:
    return ALLOWED_PLAYS.get(cause, [RETRY])


# Human-readable one-liners, surfaced in the decision trace.
CAUSE_BLURB = {
    BANK_DOWNTIME: "Issuer or PSP is failing for everyone, not just this customer.",
    SOFT_FUNDS: "Customer's account did not have the balance at that moment.",
    AUTH_FRICTION: "The customer was there but the auth step broke.",
    INSTRUMENT_DEAD: "This instrument cannot be charged again, ever.",
    RISK_BLOCK: "Declined on risk grounds. Retrying is a compliance problem.",
    TECHNICAL: "Transient error in the payment path.",
    LIMIT_EXCEEDED: "Charge exceeded a per-transaction or daily cap.",
    USER_ABANDON: "The customer walked away before completing.",
}

PLAY_BLURB = {
    RETRY: "Re-present the same instrument at a better moment",
    SWITCH_RAIL: "Move the customer to UPI instead",
    REQUEST_INSTRUMENT: "Ask for a fresh payment method",
    SUPPRESS: "Take no further attempt",
}
