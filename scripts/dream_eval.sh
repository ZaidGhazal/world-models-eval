#!/usr/bin/env bash
# Type 2 dream evaluation. Usage: scripts/dream_eval.sh [--dry-run] [dream_eval args]
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    policy="checkpoints/dry_run/policy/step_000004"
    world_model="checkpoints/dry_run/world_model/tier_1"
    if [[ ! -d "$policy" ]]; then
        policy="checkpoints/policy/smolvla_tiny/step_000200"
    fi
    if [[ ! -d "$world_model" ]]; then
        world_model="checkpoints/world_model/tiny"
    fi
    if [[ ! -d "$policy" || ! -d "$world_model" ]]; then
        echo "No dry-run/tiny policy and world-model checkpoints found; run policy and WM dry-runs first." >&2
        exit 2
    fi
    python -m dreamgrasp.eval.dream_eval --policy "$policy" --world-model "$world_model" \
        --wm-tier dry_run --n-dreams 1 --horizon 5 --episodes 1 --save-videos \
        --out results/dry_run/dream_success.parquet "$@"
    exit 0
fi

python -m dreamgrasp.eval.dream_eval "$@"
