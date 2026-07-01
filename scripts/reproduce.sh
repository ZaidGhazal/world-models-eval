#!/usr/bin/env bash
# Full Type 2 reproduction: data -> policy -> sim eval -> WM tiers -> classifier -> dreams -> study.
# Requires: GPU machine (see README), HF access, W&B login. GPU budget: ~120-180h.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m dreamgrasp.data.convert_libero                       # or reuse the published HF dataset
scripts/train_policy.sh
for ckpt in checkpoints/policy/smolvla_libero/step_*; do
    python -m dreamgrasp.eval.sim_eval --checkpoint "$ckpt" \
        --suite libero_goal --task-ids 0 1 2 3 4 5 6 7 --n-rollouts 50 --max-steps 400 --video-every 1
done
for tier in 1 2 3 4 5; do
    scripts/train_wm_tier.sh "$tier"
    python -m dreamgrasp.world_model.fidelity --checkpoint "checkpoints/world_model/tier_${tier}"
done
python -m dreamgrasp.eval.success_classifier --epochs 20
for tier in 1 2 3 4 5; do
    for ckpt in checkpoints/policy/smolvla_libero/step_*; do
        python -m dreamgrasp.eval.dream_eval --policy "$ckpt" \
            --world-model "checkpoints/world_model/tier_${tier}" --wm-tier "tier_${tier}" \
            --n-dreams 50 --horizon 200 --classifier checkpoints/classifier
    done
done
scripts/run_study.sh
