#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements.txt
python -m pytest tests -q
python scripts/run_experiment.py "$@"
python scripts/build_dashboard.py
echo
echo "Open artifacts/dashboard.html"
