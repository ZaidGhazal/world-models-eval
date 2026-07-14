#!/usr/bin/env bash
# T2.6 held-out dream rollouts: same as T2.5 but --split heldout, for the
# in-distribution vs. held-out trust-region comparison. Self-contained (activates conda,
# sets MUJOCO_GL) so it also runs correctly from a non-interactive launcher (e.g. tmux).
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
      --wm-tier "tier_${tier}" --suite libero_spatial --split heldout \
      --n-dreams 50 --horizon 200 --classifier checkpoints/classifier || rc=1
  done
done
python -m dreamgrasp.eval.acceptance dream || rc=1
echo EXIT_STATUS=$rc
echo END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
