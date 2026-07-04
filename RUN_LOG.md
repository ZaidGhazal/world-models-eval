# Type 2 Run Log

This log records real Type 2 execution evidence. Tiny and dry-run outputs do not count.

## 2026-07-03

### T2.1 Policy Training

- 2026-07-03T18:18:39Z: Initial launch attempt on `umd-004061` failed before training
  started because `scripts/train_policy.sh` lacked executable permission in the
  `v0.2-type2-ready` checkout. No W&B run was created, no checkpoint was written, and GPU
  utilization remained idle. GPU-hours consumed: 0.0.
- Fix: make shell entrypoints used by `RUNBOOK.md` executable:
  `scripts/train_policy.sh`, `scripts/train_wm_tier.sh`, `scripts/run_study.sh`, and
  `scripts/reproduce.sh`.
- The `v0.2-type2-ready` tag was moved from `e98b6066e83896c497e23d638a54288ba9887eaf`
  to the permission-fix commit so the tag resolves to a runnable Type 2 checkout.
- 2026-07-03T18:20:13Z: Relaunched real T2.1 from corrected `v0.2-type2-ready`
  (`2d56b7cc99ddc8a889872e5a792fcdb6b7c0f115`) on `umd-004061`.
  Command: `scripts/train_policy.sh`. W&B run: `ua62fo6w`
  (`smolvla_libero`). Dataset loaded: 960 train episodes, 130,386 frames.
  First logged step: step 0, loss 0.6479. Acceptance pending until 40k steps
  finish and checkpoints reload.
- While T2.1 was running, downstream launch blockers were fixed on `main` for later phases:
  checkpoint-specific simulator/dream video filenames to avoid T2.2/T2.5 overwrites, a
  classifier manifest lookup matching those filenames, and T2.4 classifier W&B logging plus
  saved confusion/metrics files with a hard failure below 90% held-out accuracy. These changes
  do not affect the active T2.1 process and will be pulled on the GPU after policy training
  finishes.
- 2026-07-03T18:37Z W&B check: run `ua62fo6w` still `running`, latest observed step 401,
  latest observed loss 0.0626, GPU utilization about 89%. No checkpoint expected yet; first
  policy checkpoint is due at step 5,000.
- 2026-07-03T19:41Z W&B/checkpoint check: run `ua62fo6w` still `running`, latest observed
  step 4,987, latest observed loss 0.0260, GPU utilization about 91%. First real policy
  checkpoint exists at `checkpoints/policy/smolvla_libero/step_005000`; keep it as the early
  low-quality rankable policy unless T2.2 shows it is not sufficiently bad.
- 2026-07-04T15:15Z: T2.1 completed. W&B run `ua62fo6w` state `finished`, `step_040000`
  checkpoint saved, `loss_first10=0.6422`, `loss_last10=0.0192`, runtime `38879.7s`
  (`10.8` GPU-hours). GPU idle afterward.
- 2026-07-04T15:27Z: T2.1 acceptance PASS. All eight saved policy checkpoints
  (`step_005000` through `step_040000`, every 5k) reloaded on CUDA with
  `dreamgrasp.eval.sim_eval.load_policy`.

### T2.2 Ground-Truth Simulator Evaluation

- 2026-07-04T15:29Z: Launched real T2.2 in tmux session `t22_sim_eval` using
  `scripts/run_t2_2_sim_eval.sh`. The wrapper runs the RUNBOOK simulator command over all
  `checkpoints/policy/smolvla_libero/step_*` checkpoints, 8 `libero_goal` task ids, 50
  rollouts, max 400 steps, and `--video-every 1`. First W&B run:
  `w8a5h3qm` (`sim_eval_smolvla_libero_step_005000`). First observed W&B rows: 3 rollouts,
  all failures at 400 steps for the early 5k checkpoint. Acceptance pending until the full
  simulator parquet is written and `python -m dreamgrasp.eval.acceptance sim` passes.

### T2.3 World-Model Family

- 2026-07-04T15:53Z: Started WM tier 1 concurrently with T2.2 in tmux session
  `t23_wm_tier1`, because T2.3 trains from the fixed LeRobot dataset and is independent of
  simulator-eval outputs. W&B run: `eyqmis9r` (`wm_tier1`). Startup check: T2.2 and T2.3
  both alive; GPU memory `5326 MiB / 24570 MiB`; GPU utilization about 97%; no OOM. WM-1
  loaded 16,272 clips and reached VAE step 600 with loss down from 0.06689 to 0.00189.
  T2.3 acceptance remains pending until the tier finishes, fidelity is computed, and the
  full tier family passes monotonicity checks.
- 2026-07-04T18:00Z: WM tier 1 training finished successfully. Artifacts written:
  `checkpoints/world_model/tier_1/vae.pt`, `dynamics.pt`, and `config.yaml`. Dynamics loss
  decreased from mean first10 `2.03667` to mean last10 `0.00338`. W&B run `eyqmis9r`
  finished and synced.
- 2026-07-04T18:03Z: WM tier 1 fidelity completed and wrote
  `results/wm_fidelity.parquet` with horizons 1, 8, 16, 32. Metrics:
  horizon 1 PSNR `26.811`, SSIM `0.943`, LPIPS `0.134`; horizon 8 PSNR `21.080`,
  SSIM `0.748`, LPIPS `0.250`; horizon 16 PSNR `19.467`, SSIM `0.692`,
  LPIPS `0.294`; horizon 32 PSNR `17.967`, SSIM `0.661`, LPIPS `0.320`;
  mean divergence step `19.8125`.
