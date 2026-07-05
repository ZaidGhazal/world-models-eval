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
- 2026-07-05T03:23Z: T2.2 job completed with `EXIT_STATUS=0` and wrote 3,200 rows to
  `results/sim_success.parquet` (8 checkpoints x 8 tasks x 50 rollouts). Runtime was about
  11.9 wall-clock GPU-hours. The acceptance check failed because checkpoint spread was below
  the required 25 points:
  `step_005000=0.0250`, `step_010000=0.0375`, `step_020000=0.1400`,
  `step_030000=0.1525`, `step_015000=0.1575`, `step_040000=0.1825`,
  `step_035000=0.1875`, `step_025000=0.2000`; best `0.2000`, worst `0.0250`,
  spread `0.1750`. T2.2 is therefore not accepted. Do not advance to T2.4/T2.5 until the
  simulator-eval gap is diagnosed; likely next checks are action normalization, task/camera
  alignment, and checkpoint-ranking assumptions.
- 2026-07-05T14:35Z: T2.2 diagnosis found a launch-scope/task-ID bug. Normalization was not
  the cause: `sim_eval.py` loaded `/home/umd-user/world-models-eval/configs/norm_stats.json`
  (`sha256=65510e254754d338121312e1a0395aa984a48a40c618a9f306fb68aa5a7a8fda`), the same
  checked-in stats used by T2.1, and an 8-step probe from `step_040000` produced normalized
  actions roughly in `[-1.01, 0.47]` with denormalized LIBERO commands inside the demo action
  range. Camera keys and resolution also matched training (`agentview`, `wrist`, 128x128).
  The real bug was that `scripts/run_t2_2_sim_eval.sh` hardcoded `libero_goal --task-ids
  0 1 2 3 4 5 6 7`. LIBERO's task-id order does not match `configs/splits.json`: those IDs
  included two held-out tasks (`put_the_wine_bottle_on_top_of_the_cabinet`,
  `turn_on_the_stove`) that had 0% success for every checkpoint, while two train tasks
  (`put_the_bowl_on_the_plate`, `put_the_wine_bottle_on_the_rack`) were never evaluated.
  It also omitted all train tasks from `libero_spatial` and `libero_object`, despite T2.1
  training on the merged three-suite train split. Filtering the existing rows to the six
  evaluated train tasks raises best success from `0.2000` to `0.2667` and spread from
  `0.1750` to `0.2333`, so the bad task selection materially depressed the gate. Bootstrap
  on the original 3,200 rows gave spread 95% CI `[0.1525, 0.2200]` and `P(spread>=0.25)
  =0.0002`; the failure is not statistical noise.
- Fix prepared on `main`: make T2.2 derive train/held-out task IDs from `configs/splits.json`,
  record `suite` and `split` in `results/sim_success.parquet`, and run acceptance on
  `--split train`. A clean corrected T2.2 rerun over 30 tasks (24 train + 6 held-out) is
  estimated at about 44-45 GPU-hours based on the previous 11.9h / 3,200-row runtime. A
  partial reuse run that keeps the already-evaluated 6 train + 2 held-out tasks and adds the
  missing 22 tasks would be about 32-33 GPU-hours, but a clean rerun is simpler and less
  error-prone for reporting.
- 2026-07-05T15:00Z: User approved the full clean corrected T2.2 rerun across all 30
  split-defined tasks, not partial reuse. Reasoning: a clean rerun avoids mixed old/new
  parquet bookkeeping and makes the report easier to audit, despite the higher estimated
  runtime (`44-45` GPU wall-clock hours). If the corrected run still misses the 25-point
  train-split spread requirement, stop and report before proposing or launching any next
  step such as increasing rollout count.

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
