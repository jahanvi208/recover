"""Build the world, train the model, run the head-to-head, write artifacts.

    python scripts/run_experiment.py

Everything downstream (the dashboard, the API) reads artifacts/ produced here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recover import nudge as nudge_mod          # noqa: E402
from recover import simulate as sim             # noqa: E402
from recover import taxonomy as tx              # noqa: E402
from recover.model import RecoveryModel         # noqa: E402
from recover.pipeline import build_world, log_exploration  # noqa: E402
from recover.policy import baseline_plan, plan_for_batch   # noqa: E402

ART = ROOT / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payments", type=int, default=24_000)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--traces", type=int, default=60, help="decision traces to export")
    ap.add_argument("--live-copy", action="store_true",
                    help="call Claude for nudge copy on exported traces")
    args = ap.parse_args()

    ART.mkdir(exist_ok=True)
    t0 = time.time()

    print("building world ...", flush=True)
    world = build_world(n_payments=args.payments)
    train_pays, test_pays = world.split(test_frac=0.25)
    print(f"  {len(world.windows)} outage windows, "
          f"{world.monitor.observed_outages} detected in telemetry")
    print(f"  {len(train_pays)} train / {len(test_pays)} test payments")

    print("logging exploration attempts ...", flush=True)
    X, y = log_exploration(train_pays, world.monitor, passes=args.passes)
    print(f"  {len(y):,} logged attempts, {y.mean():.1%} recovered")

    print("training ...", flush=True)
    model = RecoveryModel()
    report = model.fit(X, y)
    print(f"  AUC {report.auc:.4f}   Brier {report.brier:.4f}")
    model.save(ART / "model.pkl")

    print("planning on held-out payments ...", flush=True)
    pubs = [p.public() for p in test_pays]

    # Ablation ladder. Each rung adds exactly one capability, so the lift can
    # be attributed instead of asserted.
    ladder = [
        ("Learned timing only",
         dict(plays_allowed={tx.RETRY}, allow_nudge=False, allow_suppress=False)),
        ("+ rail switching",
         dict(plays_allowed=None, allow_nudge=False, allow_suppress=False)),
        ("+ nudges",
         dict(plays_allowed=None, allow_nudge=True, allow_suppress=False)),
        ("+ suppression (full agent)",
         dict(plays_allowed=None, allow_nudge=True, allow_suppress=True)),
    ]

    base_plans = [baseline_plan(p) for p in pubs]
    base_out = sim.run_policy(test_pays, base_plans,
                              "Fixed schedule (+1h/+24h/+72h)", seed=99)

    rungs = []
    agent_plans = None
    agent_out = None
    for label, kw in ladder:
        plans = plan_for_batch(pubs, model, world.monitor, **kw)
        out = sim.run_policy(test_pays, plans, label, seed=99)
        rungs.append({**out.as_dict(), **{"vs_baseline": sim.compare(out, base_out)}})
        print(f"  {label:<30} {out.recovery_rate:6.2%}  "
              f"gross ₹{out.recovered_value:>11,.0f}  "
              f"net ₹{out.net:>11,.0f}  {out.attempts:>6,} attempts")
        agent_plans, agent_out = plans, out

    delta = sim.compare(agent_out, base_out)

    print()
    print(f"  baseline : {base_out.recovery_rate:6.2%}  "
          f"net ₹{base_out.net:,.0f}  {base_out.attempts:,} attempts  "
          f"{base_out.risk_retries} risk retries")
    print(f"  agent    : {agent_out.recovery_rate:6.2%}  "
          f"net ₹{agent_out.net:,.0f}  {agent_out.attempts:,} attempts  "
          f"{agent_out.risk_retries} risk retries")
    print(f"  lift     : +{delta['recovery_rate_lift_pp']}pp  "
          f"(+{delta['value_lift_rel']}% value)  "
          f"{delta['attempts_saved_pct']}% fewer attempts  "
          f"{delta['risk_retries_avoided']} risk retries avoided")

    # ---- decision traces -------------------------------------------------
    print("exporting decision traces ...", flush=True)
    order = sorted(range(len(test_pays)), key=lambda i: -test_pays[i].amount)
    picked, seen = [], {}
    for i in order:
        c = test_pays[i].cause
        if seen.get(c, 0) < max(2, args.traces // len(tx.ROOT_CAUSES) + 1):
            seen[c] = seen.get(c, 0) + 1
            picked.append(i)
        if len(picked) >= args.traces:
            break

    traces = []
    for i in picked:
        p, plan = test_pays[i], agent_plans[i]
        st = world.monitor.state(p.bank, p.method, p.failed_at_h)
        d = plan.as_dict()
        d.update({
            "amount": p.amount,
            "method": p.method,
            "bank": p.bank,
            "network": p.network,
            "reason": p.reason,
            "is_subscription": p.is_subscription,
            "days_to_payday": p.days_to_payday,
            "rail_health": round(st.health, 3),
            "rail_degraded": st.degraded,
            "expected_recovery_h": round(st.expected_recovery_h, 2),
            "baseline_actions": [a.as_dict() for a in base_plans[i].actions],
        })
        first_nudge = next((a for a in plan.actions if a.nudge), None)
        if first_nudge is not None:
            if args.live_copy:
                copy, src = nudge_mod.write_copy(
                    p.cause, first_nudge.play, p.amount,
                    first_nudge.nudge, first_nudge.delay_h
                )
            else:
                copy = nudge_mod.offline_copy(
                    p.cause, p.amount, first_nudge.nudge, first_nudge.delay_h
                )
                src = "offline"
            d["nudge_copy"] = copy
            d["nudge_source"] = src
        traces.append(d)

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_s": round(time.time() - t0, 1),
        "world": {
            "payments": args.payments,
            "outage_windows": len(world.windows),
            "outages_detected": world.monitor.observed_outages,
            "train_payments": len(train_pays),
            "test_payments": len(test_pays),
        },
        "model": report.as_dict(),
        "agent": agent_out.as_dict(),
        "baseline": base_out.as_dict(),
        "ablation": rungs,
        "delta": delta,
        "traces": traces,
        "cause_blurbs": tx.CAUSE_BLURB,
        "play_blurbs": tx.PLAY_BLURB,
    }

    (ART / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote artifacts/results.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
