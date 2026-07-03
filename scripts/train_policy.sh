#!/usr/bin/env bash
# Type 2 policy fine-tune. Usage: scripts/train_policy.sh [--dry-run|--tiny] [extra dotlist overrides]
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    python -m dreamgrasp.policy.train --config configs/policy/smolvla_libero.yaml \
        steps=4 warmup_steps=1 batch_size=2 grad_accum=1 episodes=5 image_size=64 \
        num_workers=0 log_every=1 save_every=4 wandb=offline \
        run_name=smolvla_policy_dry_run out_dir=checkpoints/dry_run/policy "$@"
    exit 0
fi

if [[ "${1:-}" == "--tiny" ]]; then
    shift
    python -m dreamgrasp.policy.train --tiny "$@"
    exit 0
fi

python -m dreamgrasp.policy.train --config configs/policy/smolvla_libero.yaml "$@"
