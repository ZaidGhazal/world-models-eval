#!/usr/bin/env bash
# Run T2.2 simulator evaluation over all saved policy checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for ckpt in checkpoints/policy/smolvla_libero/step_*; do
    echo "CHECKPOINT=${ckpt}"
    scripts/sim_eval.sh --checkpoint "${ckpt}" \
        --suite libero_goal --task-ids 0 1 2 3 4 5 6 7 \
        --n-rollouts 50 --max-steps 400 --video-every 1
done
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
