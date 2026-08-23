"""Recovery propensity model.

A single model answers one question: given this failed payment, this play,
this delay and this nudge channel, what is the probability the money comes
back?

Everything the agent does downstream is an argmax over that function. Timing,
rail choice and channel choice are not separate models or separate rule sets;
they fall out of scoring the whole action space and picking the best expected
value.

The model is trained on logged attempts from a deliberately randomised
exploration policy, which is what makes it usable for actions the production
policy would never have tried on its own.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from .features import feature_names


@dataclass
class TrainReport:
    n_train: int
    n_test: int
    auc: float
    brier: float
    base_rate: float

    def as_dict(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 4),
            "base_rate": round(self.base_rate, 4),
        }


class RecoveryModel:
    def __init__(self) -> None:
        self.clf: CalibratedClassifierCV | None = None
        self.report: TrainReport | None = None
        self.names = feature_names()

    def fit(self, X: np.ndarray, y: np.ndarray, test_frac: float = 0.2,
            seed: int = 3) -> TrainReport:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cut = int(len(y) * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]

        base = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.07,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=seed,
        )
        # Calibration matters here: the policy compares expected rupees across
        # actions, so the probabilities have to mean what they say.
        self.clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.clf.fit(X[tr], y[tr])

        p = self.clf.predict_proba(X[te])[:, 1]
        self.report = TrainReport(
            n_train=len(tr),
            n_test=len(te),
            auc=float(roc_auc_score(y[te], p)),
            brier=float(brier_score_loss(y[te], p)),
            base_rate=float(y.mean()),
        )
        return self.report

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model is not trained. Run scripts/train.py first.")
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"clf": self.clf, "report": self.report}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "RecoveryModel":
        m = cls()
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        m.clf = blob["clf"]
        m.report = blob["report"]
        return m
