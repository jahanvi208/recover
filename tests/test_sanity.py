"""Cheap invariants that would catch a broken run before the demo does."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from recover import taxonomy as tx
from recover import world as W
from recover.bank_health import BankHealthMonitor
from recover.features import build_row, N_FEATURES
from recover.pipeline import build_world


def test_every_reason_maps_to_a_cause():
    for r in tx.REASONS:
        assert tx.classify(r) in tx.ROOT_CAUSES


def test_risk_blocks_only_allow_suppression():
    assert tx.allowed_plays(tx.RISK_BLOCK) == [tx.SUPPRESS]


def test_feature_row_width_is_stable():
    w = build_world(n_payments=50)
    row = build_row(w.payments[0].public(), tx.RETRY, 6.0, "sms", 0, w.monitor)
    assert len(row) == N_FEATURES


def test_retry_before_outage_ends_is_worse_than_after():
    p = W.Payment(
        payment_id="t", amount=5000, method="card", network="visa", bank="HDFC",
        reason="issuer_down", cause=tx.BANK_DOWNTIME, failed_at_h=100.0,
        is_subscription=False, customer_success_rate=0.8, days_to_payday=5,
        outage_ends_in_h=6.0, instrument_truly_dead=False,
        customer_reachability=0.5, upi_available=True, noise=0.0,
    )
    early = W.true_p_recover(p, tx.RETRY, 3.0, None, 0)
    late = W.true_p_recover(p, tx.RETRY, 6.5, None, 0)
    assert late > early * 3


def test_monitor_detects_a_real_outage():
    windows = W.build_downtimes(seed=11)
    mon = BankHealthMonitor(W.generate_telemetry(windows, seed=12))
    long = max(windows, key=lambda w: w.end_h - w.start_h)
    mid = (long.start_h + long.end_h) / 2
    assert mon.state(long.bank, long.method, mid).degraded


def test_agent_never_reads_the_world():
    src = (pathlib.Path(__file__).resolve().parents[1] / "recover").glob("*.py")
    for f in src:
        if f.name in {"world.py", "pipeline.py", "simulate.py", "__init__.py"}:
            continue
        assert "import world" not in f.read_text(), f"{f.name} imports the simulator"
