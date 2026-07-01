#!/usr/bin/env bash
# Type 2 policy fine-tune (real scale). Usage: scripts/train_policy.sh [extra dotlist overrides]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m dreamgrasp.policy.train --config configs/policy/smolvla_libero.yaml "$@"
