#!/usr/bin/env bash
# Regenerate the calibration study outputs (trust-region chart) from raw parquets.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m dreamgrasp.eval.correlate "$@"
