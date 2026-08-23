"""Building the dataset.

`build_world` is deterministic given a seed, so training, simulation and the
API all see the same universe.

`log_exploration` is the interesting part. The training data is not a clean
labelled set handed down from somewhere -- it is a log of attempts made by a
deliberately randomised policy, exactly like the logs a payments company
already sits on, except with the randomisation turned up so the model gets to
see actions a sensible policy would never pick. Without that exploration the
model could never learn that switching a customer to UPI at 09:00 beats a
fourth card retry at 03:00, because nobody would ever have tried it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import taxonomy as tx
from . import world as W
from .bank_health import BankHealthMonitor
from .features import build_matrix, build_row
from .policy import DELAYS_H, NUDGE_CHANNELS


@dataclass
class World:
    windows: list[W.Downtime]
    telemetry: list[dict]
    monitor: BankHealthMonitor
    payments: list[W.Payment]

    def split(self, test_frac: float = 0.25, seed: int = 5):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.payments))
        cut = int(len(self.payments) * (1 - test_frac))
        train = [self.payments[i] for i in idx[:cut]]
        test = [self.payments[i] for i in idx[cut:]]
        return train, test


def build_world(n_payments: int = 24_000, seed: int = 7) -> World:
    windows = W.build_downtimes(seed=seed + 4)
    telemetry = W.generate_telemetry(windows, seed=seed + 5)
    monitor = BankHealthMonitor(telemetry)
    payments = W.generate_payments(n_payments, windows, seed=seed)
    return World(windows, telemetry, monitor, payments)


def log_exploration(
    payments: list[W.Payment],
    monitor: BankHealthMonitor,
    passes: int = 2,
    seed: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay each payment under a random legal policy and record what happened."""
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    labels: list[int] = []

    for _ in range(passes):
        for p in payments:
            pub = p.public()
            legal = [pl for pl in tx.allowed_plays(p.cause) if pl != tx.SUPPRESS]
            if not legal:
                continue
            last = 0.0
            for attempt in range(3):
                options = [d for d in DELAYS_H if d >= last + 0.5]
                if not options:
                    break
                play = str(rng.choice(legal))
                delay = float(rng.choice(options))
                nudge = NUDGE_CHANNELS[int(rng.integers(0, len(NUDGE_CHANNELS)))]

                rows.append(build_row(pub, play, delay, nudge, attempt, monitor))
                ok, _ = W.adjudicate(rng, p, play, delay, nudge, attempt)
                labels.append(int(ok))

                if ok:
                    break
                last = delay

    return build_matrix(rows), np.asarray(labels, dtype=np.int8)
