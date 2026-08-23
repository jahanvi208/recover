"""Serving layer.

    uvicorn recover.api:app --reload

The dashboard is a static artifact, but the decision logic is real and this is
how it would be called: one failed payment in, one recovery plan out, in the
shape a webhook handler on `payment.failed` would need.

    curl -s localhost:8000/plan -H 'content-type: application/json' -d '{
      "payment_id":"pay_live_1","amount":24999,"method":"card","network":"visa",
      "bank":"HDFC","reason":"issuer_down","is_subscription":true,
      "customer_success_rate":0.82,"days_to_payday":3,"failed_at_h":181.0}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import nudge as nudge_mod
from . import taxonomy as tx
from .bank_health import BankHealthMonitor
from .model import RecoveryModel
from .policy import plan_for_batch
from .world import build_downtimes, generate_telemetry

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

app = FastAPI(title="Recover", version="0.1.0")

_state: dict[str, Any] = {}


def _boot() -> None:
    if _state:
        return
    if not (ART / "model.pkl").exists():
        raise HTTPException(503, "No trained model. Run scripts/run_experiment.py first.")
    _state["model"] = RecoveryModel.load(ART / "model.pkl")
    # Same seeds as the experiment, so health readings line up with the report.
    windows = build_downtimes(seed=11)
    _state["monitor"] = BankHealthMonitor(generate_telemetry(windows, seed=12))


class FailedPayment(BaseModel):
    payment_id: str
    amount: float = Field(gt=0)
    method: str
    bank: str
    reason: str
    network: str = "na"
    is_subscription: bool = False
    customer_success_rate: float = Field(0.75, ge=0, le=1)
    days_to_payday: int = Field(7, ge=0, le=31)
    failed_at_h: float = 100.0
    write_copy: bool = False

    def to_public(self) -> dict:
        d = self.model_dump()
        d.pop("write_copy")
        d["cause"] = tx.classify(self.reason)
        return d


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_trained": (ART / "model.pkl").exists()}


@app.get("/taxonomy")
def taxonomy() -> dict:
    return {
        "reasons": tx.REASONS,
        "root_causes": tx.ROOT_CAUSES,
        "plays": tx.PLAYS,
        "allowed_plays": tx.ALLOWED_PLAYS,
    }


@app.get("/results")
def results() -> JSONResponse:
    path = ART / "results.json"
    if not path.exists():
        raise HTTPException(404, "Run scripts/run_experiment.py first.")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    path = ART / "dashboard.html"
    if not path.exists():
        return "<pre>Run scripts/run_experiment.py then scripts/build_dashboard.py</pre>"
    return path.read_text()


@app.post("/plan")
def plan(payment: FailedPayment) -> dict:
    _boot()
    pub = payment.to_public()

    st = _state["monitor"].state(pub["bank"], pub["method"], pub["failed_at_h"])
    p = plan_for_batch([pub], _state["model"], _state["monitor"])[0]

    out = p.as_dict()
    out["rail_health"] = {
        "health": round(st.health, 3),
        "degraded": st.degraded,
        "hours_degraded": st.hours_degraded,
        "expected_recovery_h": round(st.expected_recovery_h, 2),
    }

    first = next((a for a in p.actions if a.nudge), None)
    if first is not None:
        copy, src = (
            nudge_mod.write_copy(pub["cause"], first.play, pub["amount"],
                                 first.nudge, first.delay_h)
            if payment.write_copy
            else (nudge_mod.offline_copy(pub["cause"], pub["amount"],
                                         first.nudge, first.delay_h), "offline")
        )
        out["nudge"] = {"channel": first.nudge, "copy": copy, "source": src}
    return out
