# Recover

**Track 3 — AI Revenue Recovery.** An agent that treats every failed payment as
a decision rather than a retry.

```
python -m pip install -r requirements.txt
python scripts/run_experiment.py      # ~2.5 min
python scripts/build_dashboard.py
open artifacts/dashboard.html
```

No API key needed. No network needed.

---

## The problem

A meaningful share of Indian online payments fail, and most of that money is
recoverable. What almost everybody does about it is a fixed retry schedule:
try again in an hour, then tomorrow, then in three days, same card, same rail,
until the attempts run out.

That schedule is wrong in four separate ways at once.

1. **It retries at the wrong time.** When an issuer is down, the difference
   between retrying ten minutes before it comes back and twenty minutes after
   is the difference between nothing and almost everything. A fixed schedule
   cannot see the outage, so it cannot aim at the recovery.
2. **It retries the wrong instrument.** When a customer's 3DS step times out,
   re-presenting the same card mostly reproduces the same failure. Moving them
   to UPI works.
3. **It retries things that can never work.** An expired card is dead. Every
   attempt against it is a wasted re-presentment counting against Visa's
   15-attempts-per-30-days cap, for zero possible upside.
4. **It retries things it must not touch.** Re-presenting a risk-declined
   payment is not a wasted rupee, it is chargeback and scheme-programme
   exposure.

The interesting failure mode is the fourth one, because it is the one where
optimising recovery rate makes things worse.

## The approach

Recover collapses all four into a single question, and then answers it with
one model.

> Given this failed payment, this play, this delay and this channel, what is
> the probability the money comes back?

**One model, not four.** Timing, rail choice and channel choice are not
separate systems. They are the argmax of one scored surface. Adding a new play
means adding a column, not a new service.

**A dynamic program, not a greedy pick.** Choosing the single best-scoring
action first is a trap: the top-scoring action is often at +72h, and taking it
throws away every earlier slot. Recover runs a DP over the retry calendar —
`V[slot][attempts_left]` — so the plan is optimal as a *sequence*, and it can
decide that two cheap early attempts beat one expensive late one.

**Learned from logged attempts.** The model is not trained on a clean labelled
dataset. It is trained on a log of attempts made by a deliberately randomised
policy — the same shape as the retry logs a payments company already has,
except with exploration turned up so the model gets to see actions a sensible
policy would never have tried. Without that, it could never learn that a UPI
switch at 09:00 beats a fourth card retry at 03:00, because nobody would ever
have tried it.

**Outage forecasting from telemetry.** The agent never sees outage windows. It
sees hourly success counts per bank and rail, detects degradation against a
robust baseline, fits a log-normal to observed outage durations, and computes
`E[D − e | D > e]` — how much longer this outage will probably last, given it
has already lasted `e` hours. That estimate is a feature, and it is what lets
the model aim a retry at the far side of a recovery.

**A hard compliance gate above the optimiser.** Risk declines are never
re-presented. Not because the expected value is negative — for a large enough
ticket it is positive — but because that is not a trade the optimiser is
allowed to make. This costs measurable gross recovery, and it is the right
call.

## Results

Held-out payments the model never saw. Both policies run against the same
ground-truth simulator with the same random stream, so the only difference is
the policy.

| | Fixed schedule | Recover |
|---|---|---|
| Recovery rate | 32.5% | **68.1%** |
| Net revenue recovered | ₹30.5 L | **₹78.9 L** |
| Attempts used | 20,291 | **14,763** |
| Risk declines re-presented | 755 | **0** |

Each capability, added one at a time, on the same payments:

| | Net recovered | Recovery rate |
|---|---|---|
| Fixed schedule (+1h/+24h/+72h) | ₹30.5 L | 32.5% |
| Learned timing only | ₹43.0 L | 43.3% |
| + rail switching | ₹60.9 L | 56.5% |
| + nudges | ₹74.3 L | 68.2% |
| + suppression (full agent) | **₹78.9 L** | 68.1% |

Read the last row carefully. **Suppression slightly lowers gross recovery and
raises net.** Turning it on means declining to chase money that is real but
costs more than it returns, and refusing to re-present 755 fraud declines. A
system reporting only recovery rate would have called that a regression. That
is exactly why recovery rate is the wrong headline metric for this problem.

Model: AUC 0.747, Brier 0.139, trained on 89,704 logged attempts. Probabilities
are isotonic-calibrated, because the policy compares expected rupees across
actions and the numbers have to mean what they say.

## Honest limits

- **These numbers come from a simulator, not from production.** The world model
  in `recover/world.py` is my best attempt at realistic dynamics — payday
  effects, outage recovery curves, intent decay, message fatigue — calibrated
  so a conventional fixed schedule lands in the low thirties, roughly where
  published dunning benchmarks sit. The *relative* ordering of the ablation
  rungs is the claim. The absolute lift would need a real holdout to defend.
- The agent never imports `world.py`. It learns the dynamics from logged
  attempts, which is what makes the lift meaningful rather than circular. But a
  simulator I wrote is still a simulator I wrote.
- Nudge copy quality is not measured. The uplift from messaging is modelled as
  a function of channel and reachability, not of what the message says.
- Real deployment needs per-merchant policy limits, customer-level attempt
  budgets across payments, and a live holdout arm that never gets the agent.

## Layout

```
recover/
  taxonomy.py     failure reason -> root cause -> which plays are even legal
  world.py        ground-truth simulator (the agent never imports this)
  bank_health.py  outage detection and recovery-time forecasting from telemetry
  features.py     one feature builder, shared by training and serving
  model.py        calibrated gradient boosting over the action space
  policy.py       the DP over the retry calendar, plus ablation switches
  simulate.py     head-to-head evaluation
  nudge.py        message copy (Claude when a key is present, templates otherwise)
  api.py          FastAPI: POST /plan
scripts/
  run_experiment.py   build world -> log -> train -> ablate -> write artifacts
  build_dashboard.py  bake results into one self-contained HTML file
dashboard/
  template.html       the console
```

## Serving

```bash
uvicorn recover.api:app --port 8000
curl -s localhost:8000/plan -H 'content-type: application/json' -d '{
  "payment_id":"pay_live_1","amount":24999,"method":"card","network":"visa",
  "bank":"HDFC","reason":"issuer_down","is_subscription":true,
  "customer_success_rate":0.82,"days_to_payday":3,"failed_at_h":181.0}'
```

Returns the ordered plan, the rail-health reading behind it, and the message
copy. This is the shape a `payment.failed` webhook handler would call.

Set `ANTHROPIC_API_KEY` to have Claude write the customer messages instead of
using templates; everything else is identical either way.
