"""Bank and rail health.

The agent cannot see outage windows. It sees hourly success counts per
(bank, rail) and has to work out two things:

  * is this rail degraded right now?
  * if it is, when will it come back?

The second question is the one that pays. Retrying a payment ten minutes
before an issuer comes back up burns an attempt for nothing; retrying twenty
minutes after it comes back recovers most of the money. A fixed "+1h, +24h,
+72h" schedule cannot express that.

Outage length is modelled as log-normal, fitted on degraded runs observed in
history. Remaining time is then the conditional expectation
E[D - e | D > e] for a run that has already lasted e hours.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

DEGRADED_RATIO = 0.78   # below this share of baseline, call it an outage
MIN_ATTEMPTS = 8        # below this hourly volume, the estimate is noise


@dataclass
class RailState:
    bank: str
    method: str
    health: float             # observed success rate / baseline, clipped to [0, 1.2]
    degraded: bool
    hours_degraded: float
    expected_recovery_h: float  # hours until the rail is expected to be usable

    def health_at(self, delay_h: float) -> float:
        """Projected health `delay_h` hours from now."""
        if not self.degraded:
            return min(self.health, 1.0)
        p_back = _p_recovered_by(self.hours_degraded, delay_h)
        return float(min(self.health, 1.0) * (1 - p_back) + 1.0 * p_back)


# Fitted at load time from history; these are the priors used before any fit.
_LN_MU = 0.65
_LN_SIGMA = 0.85


def _p_recovered_by(elapsed_h: float, delay_h: float) -> float:
    """P(outage ends within delay_h | it has already run elapsed_h)."""
    if delay_h <= 0:
        return 0.0
    e = max(elapsed_h, 0.05)
    s_now = 1.0 - _lognorm_cdf(e)
    s_then = 1.0 - _lognorm_cdf(e + delay_h)
    if s_now <= 1e-9:
        return 1.0
    return float(np.clip(1.0 - s_then / s_now, 0.0, 1.0))


def _lognorm_cdf(x: float) -> float:
    if x <= 0:
        return 0.0
    z = (math.log(x) - _LN_MU) / (_LN_SIGMA * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


class BankHealthMonitor:
    def __init__(self, telemetry: list[dict]):
        # (bank, method) -> hour -> ratio of successes to attempts
        self._rate: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
        self._vol: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        for r in telemetry:
            key = (r["bank"], r["method"])
            self._rate[key][r["hour"]] = r["successes"] / max(r["attempts"], 1)
            self._vol[key][r["hour"]] = r["attempts"]

        self._baseline: dict[tuple[str, str], float] = {}
        for key, series in self._rate.items():
            vals = np.array(list(series.values()))
            # Median is robust to the outages we are trying to detect.
            self._baseline[key] = float(max(np.median(vals), 0.05))

        self._fit_duration_prior()
        self._cache: dict[tuple[str, str, int], RailState] = {}

    # -- fitting ----------------------------------------------------------
    def _fit_duration_prior(self) -> None:
        global _LN_MU, _LN_SIGMA
        durations: list[float] = []
        for key, series in self._rate.items():
            base = self._baseline[key]
            hours = sorted(series)
            run = 0
            for h in hours:
                if series[h] < DEGRADED_RATIO * base:
                    run += 1
                elif run:
                    durations.append(run)
                    run = 0
            if run:
                durations.append(run)
        if len(durations) >= 20:
            logs = np.log(np.array(durations, dtype=float))
            _LN_MU = float(logs.mean())
            _LN_SIGMA = float(max(logs.std(), 0.25))
        self.observed_outages = len(durations)

    # -- query ------------------------------------------------------------
    def state(self, bank: str, method: str, t_h: float) -> RailState:
        hour = int(t_h)
        ck = (bank, method, hour)
        if ck in self._cache:
            return self._cache[ck]

        key = (bank, method)
        series = self._rate.get(key, {})
        base = self._baseline.get(key, 0.9)

        # Volume-weighted rate over a short trailing window.
        num = den = 0.0
        for h in range(hour - 2, hour + 1):
            if h in series:
                v = self._vol[key].get(h, 0)
                num += series[h] * v
                den += v
        rate = (num / den) if den >= MIN_ATTEMPTS else base
        health = float(np.clip(rate / base, 0.0, 1.2))

        degraded = health < DEGRADED_RATIO
        elapsed = 0.0
        if degraded:
            h = hour
            while h in series and series[h] < DEGRADED_RATIO * base and elapsed < 72:
                elapsed += 1
                h -= 1

        if degraded:
            # E[D - e | D > e], evaluated numerically on a fine grid.
            grid = np.arange(0.25, 48.0, 0.25)
            surv = np.array([1.0 - _lognorm_cdf(elapsed + g) for g in grid])
            s0 = 1.0 - _lognorm_cdf(max(elapsed, 0.05))
            remaining = float(np.trapezoid(surv, grid) / s0) if s0 > 1e-9 else 0.5
            remaining = float(np.clip(remaining, 0.25, 36.0))
        else:
            remaining = 0.0

        st = RailState(bank, method, health, degraded, elapsed, remaining)
        self._cache[ck] = st
        return st
