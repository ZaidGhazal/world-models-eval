#!/usr/bin/env bash
# T2.6 robustness check: T=100 vs the standard T=200, train split only (held-out ground
# truth has zero variance regardless of dream horizon -- see RUN_LOG 2026-07-14 -- so
# rerunning it here would not be informative). Self-contained for non-interactive launch.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate world-models-eval
export MUJOCO_GL=egl
cd "$(dirname "$0")/.."

echo START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
rc=0
for tier in 1 2 3 4 5; do
  for ckpt in checkpoints/policy/smolvla_libero/step_*; do
    echo "=== tier_${tier} $(basename "$ckpt") $(date -u +%H:%M:%SZ) ==="
    scripts/dream_eval.sh --policy "$ckpt" --world-model "checkpoints/world_model/tier_${tier}" \
      --wm-tier "tier_${tier}" --suite libero_spatial --split train \
      --n-dreams 50 --horizon 100 --classifier checkpoints/classifier \
      --out results/dream_success_t100.parquet --run-name "dream_eval_t100_tier_${tier}_$(basename "$ckpt")" || rc=1
  done
done
echo EXIT_STATUS=$rc
echo END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
