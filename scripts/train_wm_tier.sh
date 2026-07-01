#!/usr/bin/env bash
# Train one world-model tier. Usage: scripts/train_wm_tier.sh <1..5> [overrides]
set -euo pipefail
tier="$1"; shift
cd "$(dirname "$0")/.."
python -m dreamgrasp.world_model.train --config "configs/world_model/tier_${tier}.yaml" "$@"
