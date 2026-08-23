"""Head-to-head evaluation.

Both policies are run against the same held-out payments in the same world
with the same random stream, so the difference between them is the policy and
nothing else.

Reported per policy:
  * recovery rate and rupees recovered
  * attempts consumed (the cost side nobody reports)
  * risk-declined payments re-presented (the compliance side nobody reports)
  * net rupees after attempt and messaging cost
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import taxonomy as tx
from . import world as W
from .policy import Plan


@dataclass
class Outcome:
    label: str
    n: int = 0
    exposed: float = 0.0
    recovered_count: int = 0
    recovered_value: float = 0.0
    attempts: int = 0
    cost: float = 0.0
    risk_retries: int = 0
    by_cause: dict = field(default_factory=dict)

    @property
    def recovery_rate(self) -> float:
        return self.recovered_count / max(self.n, 1)

    @property
    def value_rate(self) -> float:
        return self.recovered_value / max(self.exposed, 1e-9)

    @property
    def net(self) -> float:
        return self.recovered_value - self.cost

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "payments": self.n,
            "exposed_value": round(self.exposed, 2),
            "recovered_count": self.recovered_count,
            "recovered_value": round(self.recovered_value, 2),
            "recovery_rate": round(self.recovery_rate, 4),
            "value_rate": round(self.value_rate, 4),
            "attempts": self.attempts,
            "attempts_per_payment": round(self.attempts / max(self.n, 1), 3),
            "cost": round(self.cost, 2),
            "net": round(self.net, 2),
            "risk_retries": self.risk_retries,
            "by_cause": {
                c: {
                    "n": v["n"],
                    "recovered": v["recovered"],
                    "rate": round(v["recovered"] / max(v["n"], 1), 4),
                    "value": round(v["value"], 2),
                }
                for c, v in sorted(self.by_cause.items())
            },
        }


def run_policy(payments: list[W.Payment], plans: list[Plan], label: str,
               seed: int = 99) -> Outcome:
    rng = np.random.default_rng(seed)
    out = Outcome(label=label)

    for pay, plan in zip(payments, plans):
        out.n += 1
        out.exposed += pay.amount
        bucket = out.by_cause.setdefault(pay.cause, {"n": 0, "recovered": 0, "value": 0.0})
        bucket["n"] += 1

        for i, act in enumerate(plan.actions):
            ok, cost = W.adjudicate(rng, pay, act.play, act.delay_h, act.nudge, i)
            out.attempts += 1
            out.cost += cost
            if pay.cause == tx.RISK_BLOCK:
                out.risk_retries += 1
            if ok:
                out.recovered_count += 1
                out.recovered_value += pay.amount
                bucket["recovered"] += 1
                bucket["value"] += pay.amount
                break

    return out


def compare(agent: Outcome, base: Outcome) -> dict:
    return {
        "recovery_rate_lift_pp": round(100 * (agent.recovery_rate - base.recovery_rate), 2),
        "recovery_rate_lift_rel": round(
            100 * (agent.recovery_rate / max(base.recovery_rate, 1e-9) - 1), 1
        ),
        "extra_value_recovered": round(agent.recovered_value - base.recovered_value, 2),
        "value_lift_rel": round(
            100 * (agent.recovered_value / max(base.recovered_value, 1e-9) - 1), 1
        ),
        "attempts_saved": base.attempts - agent.attempts,
        "attempts_saved_pct": round(
            100 * (1 - agent.attempts / max(base.attempts, 1)), 1
        ),
        "risk_retries_avoided": base.risk_retries - agent.risk_retries,
        "net_lift": round(agent.net - base.net, 2),
    }
