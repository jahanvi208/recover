"""Bake artifacts/results.json into a single self-contained HTML file.

    python scripts/build_dashboard.py

Produces artifacts/dashboard.html, which opens straight from disk with no
server and no network. That matters for a demo.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tpl = (ROOT / "dashboard" / "template.html").read_text()
results = (ROOT / "artifacts" / "results.json").read_text()

# Guard the closing script tag; the payload is inside a <script> block.
safe = results.replace("</script>", "<\\/script>")
out = ROOT / "artifacts" / "dashboard.html"
out.write_text(tpl.replace("__RESULTS_JSON__", safe))
print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size/1024:.0f} KB)")
