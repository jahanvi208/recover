"""Recovery decisioning.

Given a failed payment, search the action space and return an ordered plan of
at most three attempts. An action is a triple:

    (play, delay in hours, nudge channel)

Each candidate is scored by the model and converted to expected rupees:

    EV = P(recover) * amount - attempt_cost - nudge_cost

Attempts are chosen greedily and sequentially, because attempt two is only
reached if attempt one failed, and the model is told which attempt number it
is scoring. An attempt is only added if its expected value is positive, which
is how the policy learns to give up -- a capability a fixed retry schedule
does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import taxonomy as tx
from .bank_health import BankHealthMonitor
from .features import build_matrix, build_row
from .model import RecoveryModel

DELAYS_H = [0.25, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0]
NUDGE_CHANNELS: list[str | None] = [None, "email", "sms", "whatsapp"]
NUDGE_COST = {None: 0.0, "email": 0.02, "sms": 0.15, "whatsapp": 0.85}
ATTEMPT_COST = 1.10

MAX_ATTEMPTS = 3
MIN_SPACING_H = 0.5


@dataclass
class Action:
    play: str
    delay_h: float
    nudge: str | None
    p_recover: float
    ev: float

    def as_dict(self) -> dict:
        return {
            "play": self.play,
            "play_label": tx.PLAY_BLURB[self.play],
            "delay_h": round(self.delay_h, 2),
            "nudge": self.nudge,
            "p_recover": round(self.p_recover, 4),
            "ev": round(self.ev, 2),
        }


@dataclass
class Plan:
    payment_id: str
    cause: str
    actions: list[Action] = field(default_factory=list)
    suppressed: bool = False
    suppress_reason: str = ""
    expected_value: float = 0.0

    def as_dict(self) -> dict:
        return {
            "payment_id": self.payment_id,
            "cause": self.cause,
            "cause_blurb": tx.CAUSE_BLURB.get(self.cause, ""),
            "suppressed": self.suppressed,
            "suppress_reason": self.suppress_reason,
            "expected_value": round(self.expected_value, 2),
            "actions": [a.as_dict() for a in self.actions],
        }


def _candidates(
    cause: str,
    after_h: float,
    plays_allowed: set[str] | None,
    allow_nudge: bool,
) -> list[tuple[str, float, str | None]]:
    out = []
    for play in tx.allowed_plays(cause):
        if play == tx.SUPPRESS:
            continue
        if plays_allowed is not None and play not in plays_allowed:
            continue
        for d in DELAYS_H:
            if d < after_h + MIN_SPACING_H:
                continue
            for nudge in (NUDGE_CHANNELS if allow_nudge else [None]):
                out.append((play, d, nudge))
    # If the ablation stripped away every legal play, fall back to a plain
    # retry so the arm is still comparable rather than silently empty.
    if not out:
        for d in DELAYS_H:
            if d >= after_h + MIN_SPACING_H:
                out.append((tx.RETRY, d, None))
    return out


def _play_nudge_options(
    cause: str,
    plays_allowed: set[str] | None,
    allow_nudge: bool,
) -> list[tuple[str, str | None]]:
    plays = [
        pl for pl in tx.allowed_plays(cause)
        if pl != tx.SUPPRESS and (plays_allowed is None or pl in plays_allowed)
    ]
    if not plays:
        plays = [tx.RETRY]
    nudges = NUDGE_CHANNELS if allow_nudge else [None]
    return [(pl, nd) for pl in plays for nd in nudges]


def plan_for_batch(
    payments: list[dict],
    model: RecoveryModel,
    monitor: BankHealthMonitor,
    max_attempts: int = MAX_ATTEMPTS,
    plays_allowed: set[str] | None = None,
    allow_nudge: bool = True,
    allow_suppress: bool = True,
) -> list[Plan]:
    """Score every payment's action space in batched passes, one per slot.

    The three flags exist for ablation. Turning them off one at a time is how
    `run_experiment.py` attributes the lift to timing, rail switching, nudging
    and suppression separately, instead of reporting one number and asking to
    be believed.
    """

    plans = [Plan(p["payment_id"], p["cause"]) for p in payments]
    n_delays = len(DELAYS_H)

    # Compliance gate. Risk declines never get re-presented: the downside is
    # not a wasted rupee, it is a chargeback and a scheme penalty.
    todo = []
    for i, p in enumerate(payments):
        if allow_suppress and p["cause"] == tx.RISK_BLOCK:
            plans[i].suppressed = True
            plans[i].suppress_reason = (
                "Declined on risk grounds. Re-presenting would expose the merchant "
                "to scheme penalties, so this one is routed to manual review instead."
            )
        else:
            todo.append(i)

    if not todo:
        return plans

    # ---- score the whole calendar in one batch --------------------------
    # For every payment we need P(recover) for every (delay, play, nudge,
    # attempt number). That is a few hundred rows per payment, which is cheap,
    # and it lets the DP below see the entire action space at once.
    rows: list[list[float]] = []
    index: dict[int, list[tuple[str, str | None]]] = {}

    for i in todo:
        p = payments[i]
        acts = _play_nudge_options(p["cause"], plays_allowed, allow_nudge)
        index[i] = acts
        for d in DELAYS_H:
            for play, nudge in acts:
                for used in range(max_attempts):
                    rows.append(build_row(p, play, d, nudge, used, monitor))

    probs = model.predict(build_matrix(rows))

    # ---- dynamic program over the retry calendar ------------------------
    # V[i][a] = best expected net rupees from delay slot i onward with a
    # attempts left to spend. Taking an action at slot i pays off now with
    # probability p, and with probability (1-p) we carry on down the calendar
    # having burned one attempt.
    #
    # A greedy planner cannot do this. It will happily spend its first attempt
    # at +72h because that single action scores highest, and in doing so throw
    # away every earlier slot.
    cursor = 0
    for i in todo:
        p = payments[i]
        acts = index[i]
        n_acts = len(acts)
        block = n_delays * n_acts * max_attempts
        pr = probs[cursor:cursor + block].reshape(n_delays, n_acts, max_attempts)
        cursor += block

        amount = p["amount"]
        costs = np.array([ATTEMPT_COST + NUDGE_COST[n] for _, n in acts])

        V = np.zeros((n_delays + 1, max_attempts + 1))
        choice: dict[tuple[int, int], int] = {}

        for si in range(n_delays - 1, -1, -1):
            for a in range(1, max_attempts + 1):
                # Skipping this slot is only allowed if enough slots remain to
                # place the attempts we are still required to place. When
                # suppression is on there is no such requirement.
                can_skip = allow_suppress or (n_delays - si) > a
                best = V[si + 1][a] if can_skip else -np.inf
                pick = -1

                used = max_attempts - a
                vals = pr[si, :, used] * amount - costs + (1 - pr[si, :, used]) * V[si + 1][a - 1]
                k = int(np.argmax(vals))
                if vals[k] > best:
                    best, pick = float(vals[k]), k

                V[si][a] = best
                if pick >= 0:
                    choice[(si, a)] = pick

        # ---- reconstruct ----
        si, a = 0, max_attempts
        while si < n_delays and a > 0:
            can_skip = allow_suppress or (n_delays - si) > a
            pick = choice.get((si, a), -1)
            took = pick >= 0 and (not can_skip or V[si][a] > V[si + 1][a] + 1e-9)
            if took:
                play, nudge = acts[pick]
                prob = float(pr[si, pick, max_attempts - a])
                ev = prob * amount - (ATTEMPT_COST + NUDGE_COST[nudge])
                plans[i].actions.append(Action(play, DELAYS_H[si], nudge, prob, ev))
                a -= 1
            si += 1

        plans[i].expected_value = float(V[0][max_attempts])
        if not plans[i].actions:
            plans[i].suppressed = True
            plans[i].suppress_reason = (
                "No attempt on the calendar clears its own cost. "
                "Chasing this payment loses money."
            )

    return plans


# ---------------------------------------------------------------------------
# Baseline: what a standard dunning configuration does today
# ---------------------------------------------------------------------------

BASELINE_SCHEDULE = [1.0, 24.0, 72.0]


def baseline_plan(payment: dict) -> Plan:
    """Fixed schedule, same instrument, no nudge, no suppression.

    This is the honest comparison. It is what most retry configurations look
    like, and it is what the agent has to beat.
    """
    plan = Plan(payment["payment_id"], payment["cause"])
    for d in BASELINE_SCHEDULE:
        plan.actions.append(Action(tx.RETRY, d, None, 0.0, 0.0))
    return plan
