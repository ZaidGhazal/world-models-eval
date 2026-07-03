# Type 2 Runbook

This runbook is for the full Type 2 run after `v0.2-type2-ready`. Do not start these
full-scale commands until Prompt 2B explicitly authorizes real training/evaluation.

## Preconditions

- Checkout: `git checkout v0.2-type2-ready`
- Machine: one NVIDIA RTX 4500 Ada Generation GPU, 24,570 MiB VRAM
- Env: `conda activate world-models-eval`
- Rendering: `export MUJOCO_GL=egl`
- Dataset: `zaid9876/world-models-eval` at `v3.0`, downloaded to
  `data/lerobot/world-models-eval`
- Credentials: W&B and HF write token configured and verified
- Clean old Type 1/prepare outputs before real runs:

```bash
rm -rf checkpoints/dry_run results/dry_run
rm -f results/sim_success.parquet results/wm_fidelity.parquet results/dream_success.parquet
```

## Preflight

```bash
nvidia-smi
python -m pip check
python -m pytest
ruff check .
mypy dreamgrasp
python scripts/smoke_test.py
```

Acceptance:

- CUDA is visible to PyTorch.
- Smoke test renders LIBERO with EGL, runs SmolVLA on CUDA, and logs to W&B.
- Tests/lint/types are green.

## T2.1 Policy Training

```bash
scripts/train_policy.sh
```

Produces:

- `checkpoints/policy/smolvla_libero/step_005000`
- `checkpoints/policy/smolvla_libero/step_010000`
- `checkpoints/policy/smolvla_libero/step_020000`
- `checkpoints/policy/smolvla_libero/step_040000`
- W&B run `smolvla_libero`

Acceptance:

- Training reaches 40k steps without NaN/OOM.
- Loss decreases.
- Checkpoints reload with `dreamgrasp.eval.sim_eval.load_policy`.
- Keep an early low-quality checkpoint as the deliberately bad rankable policy.

## T2.2 Ground-Truth Simulator Evaluation

Run each rankable checkpoint over the real task set:

```bash
for ckpt in checkpoints/policy/smolvla_libero/step_*; do
  scripts/sim_eval.sh --checkpoint "$ckpt" \
    --suite libero_goal --task-ids 0 1 2 3 4 5 6 7 \
    --n-rollouts 50 --max-steps 400 --video-every 1
done
```

Produces:

- `results/sim_success.parquet`
- `results/videos/*.mp4`

Acceptance:

- Parquet columns: `checkpoint, task, seed, success, steps`.
- Wilson CIs print per task.
- Best checkpoint reaches a sane LIBERO success rate; if best is below 20%, debug
  normalization/camera/action keys before continuing.
- Spread between worst and best policy is at least 25 points.
- Add 20 categorized failure videos to `docs/failures.md`.

## T2.3 World-Model Family

```bash
for tier in 1 2 3 4 5; do
  scripts/train_wm_tier.sh "$tier"
  python -m dreamgrasp.world_model.fidelity \
    --checkpoint "checkpoints/world_model/tier_${tier}" \
    --split val --n-clips 16 --horizons 1 8 16 32
done
```

Produces:

- `checkpoints/world_model/tier_1` through `tier_5`
- `results/wm_fidelity.parquet`

Acceptance:

- Each tier trains without NaN/OOM.
- Fidelity rows exist for horizons 1, 8, 16, 32.
- WM-5 gives coherent 30-50 step rollouts at 128px.
- Fidelity is monotonic-ish across tiers. If WM-2 clearly beats WM-4, stop and debug tier design.
- Publish WM-4/WM-5 to HF only after explicit release approval.

## T2.4 Success Classifier

```bash
python -m dreamgrasp.eval.success_classifier \
  --videos-dir results/videos \
  --labels results/sim_success.parquet \
  --epochs 20 \
  --out checkpoints/classifier
```

Produces:

- `checkpoints/classifier/head.pt`
- classifier metrics in stdout/W&B

Acceptance:

- Held-out accuracy >= 90%.
- Confusion matrix is saved under `docs/`.
- Do not use the classifier for claims if it misses the 90% bar.

## T2.5 Dream Rollouts

```bash
for tier in 1 2 3 4 5; do
  for ckpt in checkpoints/policy/smolvla_libero/step_*; do
    scripts/dream_eval.sh \
      --policy "$ckpt" \
      --world-model "checkpoints/world_model/tier_${tier}" \
      --wm-tier "tier_${tier}" \
      --n-dreams 50 \
      --horizon 200 \
      --classifier checkpoints/classifier
  done
done
```

Produces:

- `results/dream_success.parquet`
- optional dream videos when `--save-videos` is used

Acceptance:

- Parquet columns: `checkpoint, task, wm_tier, seed, dream_success_prob`.
- One `(checkpoint, task, tier)` triple evaluates in under 10 minutes.
- Dream videos are not used for claims until the classifier has passed T2.4.

## T2.6 Calibration Study

```bash
scripts/run_study.sh
```

Produces:

- `report/figures/trust_region.png`
- reliability/fidelity summary in stdout

Acceptance:

- Spearman rho with bootstrap CIs per tier.
- Task-level Pearson for the best policy.
- Held-out task results are included.
- Robustness includes at least two: N=20 vs 50, classifier threshold +/-0.1, T=100 vs 200.
- `LIMITATIONS.md` covers sim-only, small models, classifier bound, single embodiment,
  and LIBERO-specific scope.

## T2.7 Release Prep

Do not publish release artifacts without explicit approval.

```bash
scripts/make_gif.py --sim <sim_video.mp4> --dream <dream_video.mp4> --out report/figures/money.gif
python space/app.py
```

Produces:

- money GIF
- final report figures
- precomputed Space assets
- HF artifacts: `world-models-eval-vla`, `world-models-eval-worldmodel`,
  `world-models-eval-demo`

Acceptance:

- README leads with the GIF and trust-region chart.
- Report cites WorldEval, WPE, Ctrl-World, and SIMPLER in the intro.
- HF artifacts are cross-linked.
- CI green.
- `CITATION.cff` and Apache-2.0 license present.

## T2.8 Final Reproduction

```bash
bash scripts/reproduce.sh
```

Acceptance:

- A fresh clone with GPU access can regenerate the trust-region chart from raw parquets.
- All four HF artifacts are live and cross-linked.
- Tag `v1.0` only after explicit approval.

## Cut Order

Use this order only if the full plan exceeds VRAM or takes too long:

1. Train 3 WM tiers only: 10%, 50%, 100%.
2. Reduce world-model image/latent resolution to 64px.
3. Use 2 task suites instead of 3.
4. Reduce rollout count from N=50 to N=30.

Never cut fixed splits, confidence intervals, or held-out task evaluation.

## Wall-Clock Estimate

| Phase | Estimate |
|---|---:|
| Policy fine-tune + simulator evaluation | 40-60 h |
| World-model family, 5 tiers | 50-80 h |
| Success classifier | 5 h |
| Dream rollouts | 20-30 h |
| Analysis/release assembly | 5 h |
| Total occupied GPU wall-clock | 120-180 h |

No dollar cost is estimated because this is owned hardware, not a billed cloud instance.
