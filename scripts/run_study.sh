#!/usr/bin/env bash
# Regenerate calibration outputs. Usage: scripts/run_study.sh [--dry-run|correlate args]
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    python -m dreamgrasp.eval.correlate --synthetic 0.95 "$@"
    python -m dreamgrasp.eval.correlate --synthetic 0.0 "$@"
    exit 0
fi

python -m dreamgrasp.eval.correlate "$@"
