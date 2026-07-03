#!/usr/bin/env bash
# Type 2 simulator evaluation. Usage: scripts/sim_eval.sh [--dry-run] [sim_eval args]
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    ckpt="checkpoints/dry_run/policy/step_000020"
    if [[ ! -d "$ckpt" ]]; then
        ckpt="checkpoints/policy/smolvla_tiny/step_000200"
    fi
    if [[ ! -d "$ckpt" ]]; then
        echo "No dry-run or tiny policy checkpoint found; run scripts/train_policy.sh --dry-run first." >&2
        exit 2
    fi
    python -m dreamgrasp.eval.sim_eval --checkpoint "$ckpt" --suite libero_goal \
        --task-ids 0 --n-rollouts 1 --max-steps 5 --video-every 1 \
        --out results/dry_run/sim_success.parquet "$@"
    exit 0
fi

python -m dreamgrasp.eval.sim_eval "$@"
