#!/usr/bin/env bash
# Train one world-model tier. Usage: scripts/train_wm_tier.sh <1..5> [--dry-run|overrides]
set -euo pipefail
tier="$1"; shift
cd "$(dirname "$0")/.."

if [[ ! "$tier" =~ ^[1-5]$ ]]; then
    echo "usage: scripts/train_wm_tier.sh <1..5> [--dry-run|overrides]" >&2
    exit 2
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    python -m dreamgrasp.world_model.train --config "configs/world_model/tier_${tier}.yaml" \
        episodes=5 data_fraction=1.0 vae_steps=20 dyn_steps=20 batch_size=2 \
        num_workers=0 log_every=5 save_every=20 wandb=offline \
        name="wm_tier${tier}_dry_run" out_dir="checkpoints/dry_run/world_model/tier_${tier}" "$@"
    exit 0
fi

python -m dreamgrasp.world_model.train --config "configs/world_model/tier_${tier}.yaml" "$@"
