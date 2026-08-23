"""The world.

This module is the *ground truth simulator*, and the agent never imports it.
It plays two roles:

1. It generates a stream of failed payments with hidden state (when an outage
   actually ends, when a customer actually gets paid, whether a card is truly
   dead).
2. It adjudicates a recovery attempt: given an action, did the money come back?

Keeping this strictly separate from `policy` and `model` is the whole point.
The agent has to *learn* the dynamics below from logged attempts; it is never
handed the formula. That is what makes the measured lift in `simulate.py`
mean something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import taxonomy as tx

# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PNB", "YES", "IDFC", "BOB", "CANARA"]
METHODS = ["card", "upi", "netbanking", "wallet"]
NETWORKS = ["visa", "mastercard", "rupay", "amex", "na"]

METHOD_MIX = np.array([0.34, 0.47, 0.12, 0.07])

# Which failure reasons show up on which rail, and how often.
REASON_MIX = {
    "card": {
        "insufficient_funds": 0.16,
        "payment_authentication_failed": 0.14,
        "incorrect_otp": 0.11,
        "otp_attempts_exceeded": 0.05,
        "three_ds_failed": 0.06,
        "card_expired": 0.07,
        "card_blocked": 0.05,
        "issuer_down": 0.11,
        "gateway_technical_error": 0.06,
        "payment_declined_by_risk": 0.05,
        "suspected_fraud": 0.02,
        "server_error": 0.04,
        "per_transaction_limit_exceeded": 0.04,
        "international_transaction_not_allowed": 0.02,
        "customer_cancelled": 0.02,
    },
    "upi": {
        "upi_collect_expired": 0.24,
        "insufficient_funds": 0.18,
        "upi_psp_down": 0.13,
        "npci_unavailable": 0.08,
        "invalid_vpa": 0.06,
        "payment_timed_out": 0.09,
        "daily_limit_exceeded": 0.06,
        "customer_cancelled": 0.07,
        "account_balance_low": 0.05,
        "server_error": 0.03,
        "payment_declined_by_risk": 0.01,
    },
    "netbanking": {
        "issuer_down": 0.24,
        "payment_authentication_failed": 0.16,
        "insufficient_funds": 0.15,
        "server_error": 0.11,
        "payment_timed_out": 0.12,
        "customer_cancelled": 0.10,
        "daily_limit_exceeded": 0.07,
        "account_closed": 0.05,
    },
    "wallet": {
        "account_balance_low": 0.34,
        "payment_timed_out": 0.16,
        "customer_cancelled": 0.16,
        "server_error": 0.12,
        "issuer_down": 0.10,
        "daily_limit_exceeded": 0.07,
        "payment_declined_by_risk": 0.05,
    },
}

RETRY_OFFSETS_H = [0.25, 1.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0, 72.0]
NUDGE_CHANNELS = [None, "email", "sms", "whatsapp"]
NUDGE_COST = {None: 0.0, "email": 0.02, "sms": 0.15, "whatsapp": 0.85}
# Cost of a declined re-presentment: gateway fee plus amortised exposure to
# network excessive-retry programmes.
ATTEMPT_COST = 1.10
# Expected cost of re-presenting a risk-declined payment: P(chargeback) times
# representment fee, plus amortised exposure to Visa VAMP / Mastercard ECP
# thresholds. Cheap per attempt, ruinous in aggregate.
RISK_RETRY_PENALTY = 600.0


@dataclass
class Payment:
    """A failed payment plus the hidden state that decides its fate."""

    payment_id: str
    amount: float
    method: str
    network: str
    bank: str
    reason: str
    cause: str
    failed_at_h: float          # hours since epoch of the simulation
    is_subscription: bool
    customer_success_rate: float
    days_to_payday: int
    # --- hidden ---
    outage_ends_in_h: float = 0.0
    instrument_truly_dead: bool = True
    customer_reachability: float = 0.5
    upi_available: bool = True
    noise: float = field(default=0.0)

    def public(self) -> dict:
        """Exactly what the agent is allowed to see."""
        return {
            "payment_id": self.payment_id,
            "amount": self.amount,
            "method": self.method,
            "network": self.network,
            "bank": self.bank,
            "reason": self.reason,
            "cause": self.cause,
            "failed_at_h": self.failed_at_h,
            "is_subscription": self.is_subscription,
            "customer_success_rate": self.customer_success_rate,
            "days_to_payday": self.days_to_payday,
        }


@dataclass
class Downtime:
    bank: str
    method: str
    start_h: float
    end_h: float
    severity: float  # fraction of success rate wiped out

    def active(self, t: float) -> bool:
        return self.start_h <= t < self.end_h


HORIZON_H = 24 * 30
BASELINE_SUCCESS = 0.91


def build_downtimes(seed: int = 11, horizon_h: float = HORIZON_H) -> list[Downtime]:
    """Latent outage windows. Both the telemetry stream and individual payment
    failures are generated from these, which is why bank health has real
    predictive content rather than being decoration."""
    rng = np.random.default_rng(seed)
    windows: list[Downtime] = []
    for bank in BANKS:
        for method in METHODS:
            # ~1 outage per bank/rail per 4 days, heavier on netbanking.
            rate = {"card": 0.010, "upi": 0.008, "netbanking": 0.016, "wallet": 0.007}[method]
            n = rng.poisson(rate * horizon_h)
            for _ in range(int(n)):
                start = float(rng.uniform(0, horizon_h))
                dur = float(np.clip(rng.lognormal(0.6, 0.8), 0.3, 14.0))
                windows.append(
                    Downtime(bank, method, start, start + dur, float(rng.uniform(0.45, 0.95)))
                )
    return windows


def active_downtime(windows: list[Downtime], bank: str, method: str, t: float) -> Downtime | None:
    for w in windows:
        if w.bank == bank and w.method == method and w.active(t):
            return w
    return None


def generate_telemetry(windows: list[Downtime], seed: int = 12,
                       horizon_h: float = HORIZON_H) -> list[dict]:
    """Hourly platform-wide success counts per bank and rail.

    This is the only outage signal the agent gets. It has to infer "this bank
    is down and will probably be back around 03:40" from these counts.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for hour in range(int(horizon_h)):
        for bank in BANKS:
            for method in METHODS:
                volume = int(rng.poisson({"card": 120, "upi": 260, "netbanking": 45, "wallet": 25}[method]))
                if volume == 0:
                    continue
                rate = BASELINE_SUCCESS
                w = active_downtime(windows, bank, method, hour + 0.5)
                if w is not None:
                    rate *= (1.0 - w.severity)
                success = int(rng.binomial(volume, np.clip(rate, 0.01, 0.99)))
                rows.append(
                    {"hour": hour, "bank": bank, "method": method,
                     "attempts": volume, "successes": success}
                )
    return rows


def _sample_reason(rng: np.random.Generator, method: str) -> str:
    mix = REASON_MIX[method]
    keys = list(mix)
    probs = np.array([mix[k] for k in keys], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(keys, p=probs))


def generate_payments(n: int, windows: list[Downtime], seed: int = 7,
                      horizon_h: float = HORIZON_H) -> list[Payment]:
    rng = np.random.default_rng(seed)
    out: list[Payment] = []

    for i in range(n):
        method = str(rng.choice(METHODS, p=METHOD_MIX))
        bank = str(rng.choice(BANKS))
        failed_at = float(rng.uniform(0, horizon_h))

        w = active_downtime(windows, bank, method, failed_at)
        if w is not None and rng.random() < 0.55 + 0.4 * w.severity:
            reason = "upi_psp_down" if method == "upi" else "issuer_down"
        else:
            reason = _sample_reason(rng, method)
        cause = tx.classify(reason)

        # Log-normal ticket sizes: lots of small UPI, a long tail of big cards.
        base = {"card": 7.2, "upi": 6.2, "netbanking": 8.0, "wallet": 5.6}[method]
        amount = float(np.clip(rng.lognormal(base, 0.85), 20, 400_000))

        network = str(rng.choice(NETWORKS[:4])) if method == "card" else "na"

        p = Payment(
            payment_id=f"pay_{i:07d}",
            amount=round(amount, 2),
            method=method,
            network=network,
            bank=bank,
            reason=reason,
            cause=cause,
            failed_at_h=failed_at,
            is_subscription=bool(rng.random() < 0.22),
            customer_success_rate=float(np.clip(rng.beta(6, 2), 0.05, 0.99)),
            days_to_payday=int(rng.integers(0, 30)),
            outage_ends_in_h=(
                max(0.05, w.end_h - failed_at)
                if (cause == tx.BANK_DOWNTIME and w is not None)
                else (float(abs(rng.normal(1.2, 1.0))) if cause == tx.BANK_DOWNTIME else 0.0)
            ),
            # "card_expired" is always terminal; "card_blocked" sometimes is a
            # temporary freeze the customer can lift.
            instrument_truly_dead=(reason in {"card_expired", "account_closed"}) or rng.random() < 0.62,
            customer_reachability=float(np.clip(rng.beta(3, 3), 0.02, 0.98)),
            upi_available=bool(rng.random() < 0.86),
            noise=float(rng.normal(0, 0.35)),
        )
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# True recovery probability
# ---------------------------------------------------------------------------

def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def true_p_recover(
    p: Payment,
    play: str,
    delay_h: float,
    nudge: str | None,
    attempt_index: int,
) -> float:
    """Ground truth. The agent must approximate this from data alone."""

    if play == tx.SUPPRESS:
        return 0.0

    # Intercept calibrated so that a conventional fixed retry schedule lands
    # in the high-twenties recovery rate, which is roughly where published
    # involuntary-churn dunning benchmarks sit.
    z = -2.05
    # Fatigue: every extra ask at the same customer converts worse.
    z += -0.45 * attempt_index

    # Bigger tickets are harder to recover across the board.
    z += -0.28 * (math.log10(max(p.amount, 1.0)) - 3.0)
    z += 1.05 * (p.customer_success_rate - 0.7)

    local_hour = (p.failed_at_h + delay_h) % 24
    # Nobody authorises a payment at 3am.
    awake = _sig(3.0 * math.sin((local_hour - 8.5) / 24 * 2 * math.pi) + 1.4)

    if p.cause == tx.BANK_DOWNTIME:
        if play == tx.SWITCH_RAIL:
            z += 1.30 if p.upi_available else -2.2
        else:
            # Retry before the outage clears and you get nothing; retry just
            # after and you get almost all of it back.
            if delay_h < p.outage_ends_in_h:
                z += -3.4
            else:
                z += 2.15 - 0.012 * (delay_h - p.outage_ends_in_h)

    elif p.cause == tx.SOFT_FUNDS:
        if play == tx.REQUEST_INSTRUMENT:
            z += -0.25 + 1.4 * p.customer_reachability
        else:
            # Balance arrives on payday. This is the single strongest timing
            # signal in Indian recurring payments and a fixed +24h schedule
            # cannot see it.
            days_out = delay_h / 24.0
            gap = abs(p.days_to_payday - days_out)
            z += 2.4 * math.exp(-0.9 * gap) - 1.5
            z += -0.35 * math.log10(max(p.amount, 1.0))

    elif p.cause == tx.AUTH_FRICTION:
        if play == tx.SWITCH_RAIL:
            z += 1.55 if p.upi_available else -2.0
        elif play == tx.REQUEST_INSTRUMENT:
            z += 0.15 + 1.1 * p.customer_reachability
        else:
            # Same broken 3DS flow, mostly breaks the same way again.
            z += 0.30 - 0.004 * delay_h

    elif p.cause == tx.INSTRUMENT_DEAD:
        if p.instrument_truly_dead:
            if play == tx.REQUEST_INSTRUMENT:
                z += -0.9 + 2.3 * p.customer_reachability
            elif play == tx.SWITCH_RAIL:
                z += -0.4 + 1.2 * p.customer_reachability
            else:
                return 0.004
        else:
            z += 0.9 if play != tx.RETRY else 0.1

    elif p.cause == tx.RISK_BLOCK:
        return 0.015

    elif p.cause == tx.TECHNICAL:
        # Free money, but only if you move fast.
        z += 2.5 - 0.9 * math.log1p(delay_h)

    elif p.cause == tx.LIMIT_EXCEEDED:
        if play == tx.SWITCH_RAIL:
            z += 0.85 if p.upi_available else -2.0
        else:
            # Daily caps reset at midnight.
            crossed_midnight = (p.failed_at_h % 24) + delay_h >= 24
            z += 1.9 if crossed_midnight else -2.1

    elif p.cause == tx.USER_ABANDON:
        # Intent decays fast. Catch them in the first hour or lose them.
        z += 1.45 * math.exp(-delay_h / 9.0) - 0.5

    # Nudges only help when a human has to do something.
    if nudge is not None:
        human_in_loop = p.cause in {
            tx.SOFT_FUNDS, tx.INSTRUMENT_DEAD, tx.USER_ABANDON, tx.AUTH_FRICTION
        } or play in {tx.REQUEST_INSTRUMENT, tx.SWITCH_RAIL}
        strength = {"email": 0.22, "sms": 0.45, "whatsapp": 0.80}[nudge]
        z += strength * (1.6 * p.customer_reachability if human_in_loop else 0.10)

    z += 0.55 * (awake - 0.5)
    z += p.noise

    return float(np.clip(_sig(z), 0.001, 0.985))


def adjudicate(
    rng: np.random.Generator,
    p: Payment,
    play: str,
    delay_h: float,
    nudge: str | None,
    attempt_index: int,
) -> tuple[bool, float]:
    """Run one attempt. Returns (recovered, cost_incurred)."""
    if play == tx.SUPPRESS:
        return False, 0.0

    cost = NUDGE_COST[nudge] + ATTEMPT_COST
    if p.cause == tx.RISK_BLOCK:
        cost += RISK_RETRY_PENALTY

    prob = true_p_recover(p, play, delay_h, nudge, attempt_index)
    return bool(rng.random() < prob), cost
