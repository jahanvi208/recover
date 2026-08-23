"""Nudge copy.

The policy decides *whether* to message a customer, on *which* channel, and
*when*. This module decides what the message says.

Claude writes it when an API key is present. There is a deterministic
fallback so the demo never depends on the network, and so the copy is
reproducible in the write-up.

The one rule the generated copy has to respect: never tell the customer their
payment failed for a reason we are not sure about. The root cause is a
classification, not a fact, and a wrong explanation costs more trust than a
vague one.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import taxonomy as tx

MODEL = "claude-sonnet-4-6"
LIMITS = {"sms": 160, "whatsapp": 320, "email": 400}

FALLBACK = {
    tx.SOFT_FUNDS: "Your payment of {amt} to {m} didn't go through. We'll try again on {when} — no action needed if your balance is topped up by then.",
    tx.BANK_DOWNTIME: "Your bank was briefly unavailable, so your {amt} payment to {m} didn't complete. We're retrying automatically once it's back up.",
    tx.AUTH_FRICTION: "The verification step timed out on your {amt} payment to {m}. Paying by UPI takes about ten seconds: {link}",
    tx.INSTRUMENT_DEAD: "We couldn't charge your saved card for {amt} to {m}. Add another payment method here: {link}",
    tx.USER_ABANDON: "Your {amt} payment to {m} is still waiting. Finish it here: {link}",
    tx.LIMIT_EXCEEDED: "Your {amt} payment to {m} went over a bank limit. We'll retry after the limit resets, or you can pay now by UPI: {link}",
    tx.TECHNICAL: "Something went wrong on our side with your {amt} payment to {m}. We're retrying it automatically.",
}


def _fmt_inr(amount: float) -> str:
    """Indian digit grouping."""
    n = int(round(amount))
    s = str(n)
    if len(s) <= 3:
        return f"₹{s}"
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "₹" + ",".join(parts) + "," + tail


def _when(delay_h: float) -> str:
    if delay_h < 1:
        return "in a few minutes"
    if delay_h < 24:
        return f"in about {int(round(delay_h))} hours"
    days = delay_h / 24
    return "tomorrow" if days < 1.6 else f"in {int(round(days))} days"


def offline_copy(cause: str, amount: float, channel: str, delay_h: float,
                 merchant: str = "your order") -> str:
    tpl = FALLBACK.get(cause, FALLBACK[tx.TECHNICAL])
    text = tpl.format(
        amt=_fmt_inr(amount), m=merchant, when=_when(delay_h), link="rzp.io/r/xxxx"
    )
    return text[: LIMITS.get(channel, 320)]


def write_copy(cause: str, play: str, amount: float, channel: str,
               delay_h: float, merchant: str = "your order",
               timeout: float = 20.0) -> tuple[str, str]:
    """Returns (copy, source) where source is 'claude' or 'offline'."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return offline_copy(cause, amount, channel, delay_h, merchant), "offline"

    prompt = f"""Write one recovery message to an Indian customer whose payment failed.

Channel: {channel} (hard limit {LIMITS.get(channel, 320)} characters)
Amount: {_fmt_inr(amount)} to {merchant}
What our classifier believes went wrong: {tx.CAUSE_BLURB.get(cause, 'Unknown')}
What we are about to do: {tx.PLAY_BLURB.get(play, 'retry')}, {_when(delay_h)}

Rules:
- Plain Indian English. No exclamation marks, no apology theatre.
- Do not state the failure reason as certain. Our classification can be wrong.
- Say exactly what happens next and what, if anything, they need to do.
- If they need to act, end with the link placeholder rzp.io/r/xxxx
- Output only the message text."""

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        text = text.strip()
        if text:
            return text[: LIMITS.get(channel, 320)], "claude"
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        pass
    return offline_copy(cause, amount, channel, delay_h, merchant), "offline"
