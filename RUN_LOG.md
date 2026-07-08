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
- 2026-07-05T15:28Z: First clean-rerun launch attempt exited after 3 seconds before any
  rollout work. Cause: LIBERO printed its task-order info line to stdout while
  `scripts/run_t2_2_sim_eval.sh` was generating the tab-separated task plan, so the shell
  parsed that info line as a suite and `sim_eval.py` rejected an empty `--task-ids` list.
  GPU-hours consumed: effectively 0.0. Fix: redirect LIBERO task-order stdout to stderr
  during task-plan generation, then relaunch cleanly.
- 2026-07-05T15:29Z: Relaunched the approved clean T2.2 rerun in tmux session
  `t22_sim_eval_clean` from commit `402ef702e7791c92ca97a5399c70f5f9de07962d`.
  Previous failed T2.2 artifacts were archived under
  `results/archive/t2_2_failed_20260705_20260705T152816Z`, and the 3-second failed-launch
  log was preserved as `logs/t2.2_clean_rerun_launch_failed_20260705T1528.log`. The clean
  run is writing to `logs/t2.2_clean_rerun.log`; first W&B run is `3et5zrab`
  (`sim_eval_smolvla_libero_step_005000`) for `libero_spatial/train` with task IDs
  `0 2 4 6 8 1 3 5`. Acceptance remains pending until the full 30-task rerun finishes and
  `python -m dreamgrasp.eval.acceptance sim --split train` passes.
- 2026-07-07T18:33Z: Clean T2.2 rerun completed all 12,000 rollouts and wrote 12,000 rows
  to `results/sim_success.parquet` (24 train tasks + 6 held-out tasks, 8 checkpoints, 50
  rollouts per checkpoint/task). The final acceptance command failed the train-split spread
  gate: `step_010000=0.0275`, `step_005000=0.0283`, `step_020000=0.1433`,
  `step_030000=0.1758`, `step_015000=0.1883`, `step_035000=0.1975`,
  `step_040000=0.1975`, `step_025000=0.2083`; best `0.2083`, worst `0.0275`, spread
  `0.1808` (< required `0.2500`). Tmux exited with `EXIT_STATUS=1` due to the failed
  acceptance check, not a runtime crash. Per user instruction, stop here and report before
  proposing or launching any next step.
- 2026-07-08: T2.2 scope decision approved by the user, acting on the DIAGNOSTIC-2 per-suite
  analysis of the completed 12,000-row `results/sim_success.parquet`: the T2.2 gate for
  proceeding to T2.4/T2.5/T2.6 is redefined to the `libero_spatial` subset of the train
  split, where worst-to-best checkpoint spread is `27.00` points (clears the 25-point bar).
  `libero_object` (`23.5`) and `libero_goal` (`16.75`) did not reach usable spread within
  this run's 40k-step training budget, even after removing their hardest tasks. No rerun is
  needed or permitted to act on this decision; the existing parquet is the input, and
  object/goal rows stay in the parquet and in the final report as non-gating results.
  Implemented in commit `5254820a88a6778375ede8e525411bc649bf494e`: `acceptance.py` gains a
  `--suite` filter that prints full-scope (all-suite) numbers as non-gating before gating on
  the suite subset, with a unit test (`tests/test_acceptance.py`), and RUNBOOK.md/
  LIMITATIONS.md now document the scope. New acceptance command:
  `python -m dreamgrasp.eval.acceptance sim --split train --suite libero_spatial`.
  GPU-hours consumed: 0.0 (documentation/configuration change only; verified with pytest,
  ruff, mypy, and a synthetic-parquet CLI run on the Mac).
- 2026-07-08 open items from this session, blocked on GPU-host access (`umd-004061` SSH is
  password-only and this session could not authenticate non-interactively): (1) the free
  video inspection of failed `libero_object`/`libero_goal` rollouts — the saved
  `results/videos/*.mp4` exist only on the GPU host and sim-eval does not upload videos to
  W&B — so the failure-mode sentence in LIMITATIONS.md item 6 is marked TODO; (2) the
  authoritative acceptance re-run against the real 12,000-row parquet, which must be executed
  on the GPU host after pulling `main`. Both take minutes and 0 GPU-hours. Do not start T2.4
  until the acceptance command passes on the GPU host and the video-inspection sentence lands.
- 2026-07-08 (later): user provided SSH access to `umd-004061`; both open items are resolved.
  Commits were synced to the GPU via branch `t2.2-spatial-scope` (direct push to `main` was
  blocked by session policy; the branch should be merged/fast-forwarded to `main` by the
  user). GPU checkout at `db6d014`. GPU-hours consumed: 0.0 (CPU-only parquet read and
  video decoding of existing artifacts).
  (1) Authoritative acceptance PASS on the real 12,000-row `results/sim_success.parquet`:
  `python -m dreamgrasp.eval.acceptance sim --split train --suite libero_spatial` exited 0.
  Gating subset (libero_spatial train, 3,200 rows): `step_010000=0.0200`,
  `step_005000=0.0475`, `step_020000=0.1575`, `step_015000=0.1975`, `step_040000=0.2325`,
  `step_030000=0.2450`, `step_035000=0.2525`, `step_025000=0.2900`; best `0.290`, spread
  `0.270` (>= `0.250`). Full-scope non-gating numbers printed by the same command (9,600
  train rows): best `0.208`, spread `0.181`, matching the 2026-07-07 failure entry.
  (2) Video inspection of 18 failed `libero_object`/`libero_goal` train rollouts (6 tasks x
  checkpoints 5k/20k/40k, sampled from the parquet, copied from the GPU and reviewed as
  12-frame contact sheets): early checkpoints sweep the arm over the workspace without
  contacting the task object; mid/late checkpoints reach the correct region but hover above
  the object, close the gripper on nothing, and then stall or move to the goal location
  (basket/stove/drawer) empty-handed; one case engaged the correct object but dragged it
  past the target (plate pulled off the counter, `push_the_plate` at 20k); one near-miss
  carried an object to the basket rim without ending inside (`alphabet_soup` at 20k). All
  sampled failures ran the full 400 steps; no dropped-object or scene-destroying behavior.
  Findings recorded in `LIMITATIONS.md` item 6. T2.2 is now accepted under the
  libero_spatial scope; stopping here for approval before starting T2.4.

### T2.4 Success Classifier

- 2026-07-08T04:45Z (approx): Pre-launch items resolved. (1) Branch `t2.2-spatial-scope`
  was fast-forwarded into `main` (`52cf9b6..0e59a58`, no merge commit needed since the
  branch was strictly ahead) and the GPU checkout switched back to `main` at `0e59a58`;
  the remote feature branch was left in place. (2) T2.4 training-data scope confirmed as
  all three suites (full 12,000-video `results/sim_success.parquet` manifest, train and
  held-out splits): the classifier is a success judge, not the ranked policy, so its
  training scope need not match T2.6's libero_spatial ranking scope, and the diverse
  examples cost nothing. `success_classifier.build_manifest` already uses every parquet
  row with an existing video (no suite filter), so no code change was needed; RUNBOOK.md
  T2.4 now states the scope explicitly (commit `6bcee82`).
- 2026-07-08T04:48Z: Launched real T2.4 in tmux session `t24_classifier` on `umd-004061`
  from `main` at `6bcee82`, GPU idle beforehand (155 MiB used). Command per RUNBOOK:
  `python -m dreamgrasp.eval.success_classifier --videos-dir results/videos --labels
  results/sim_success.parquet --epochs 20 --out checkpoints/classifier`, logging to
  `logs/t2.4_classifier.log`. W&B run: `nri5o6vs` (`success_classifier`). Startup check:
  session alive, GPU at 901 MiB / 15% utilization during the frozen-SigLIP pre-embedding
  pass over 12,000 videos. Acceptance pending: held-out accuracy >= 90% (hard failure in
  the script below that bar), confusion matrix under `docs/`.

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
- 2026-07-05T16:14Z: Started WM tier 2 concurrently with the clean T2.2 rerun in tmux
  session `t23_wm_tier2`, because T2.3 trains from the fixed LeRobot dataset and remains
  independent of simulator-eval outputs. Command sequence: `scripts/train_wm_tier.sh 2`,
  then `python -m dreamgrasp.world_model.fidelity --checkpoint checkpoints/world_model/tier_2
  --split val --n-clips 16 --horizons 1 8 16 32` only if training exits successfully. W&B
  run: `rsa2f1vy` (`wm_tier2`). Startup check: T2.2 and T2.3 both alive; T2.2 was still
  writing rollout lines from `libero_spatial/train`, tier 2 loaded 31,447 clips, VAE loss
  decreased from `0.06652` at step 0 to `0.00115` by step 1100, and GPU memory remained
  `5404 MiB / 24570 MiB` with temperature `73C`. No OOM or resource contention observed.
  T2.3 tier 2 acceptance remains pending until training and fidelity complete.
- 2026-07-05T17:48Z: WM tier 2 training and fidelity completed successfully. Fidelity wrote
  `results/wm_fidelity.parquet` rows for tier 2: horizon 1 PSNR `27.041`, SSIM `0.939`,
  LPIPS `0.132`; horizon 8 PSNR `21.626`, SSIM `0.752`, LPIPS `0.245`; horizon 16 PSNR
  `20.276`, SSIM `0.721`, LPIPS `0.272`; horizon 32 PSNR `18.649`, SSIM `0.698`,
  LPIPS `0.306`; mean divergence step `20.9375`.
- 2026-07-05T22:38Z: Started WM tier 3 concurrently with the clean T2.2 rerun in tmux
  session `t23_wm_tier3` after confirming tier 2 had finished and GPU memory was low
  (`1357 MiB / 24570 MiB`). Command sequence mirrors tier 2: `scripts/train_wm_tier.sh 3`,
  then tier 3 fidelity if training exits successfully. W&B run: `eibpvamc` (`wm_tier3`).
  Startup check: T2.2 and T2.3 both alive; tier 3 loaded 66,116 clips, VAE loss decreased
  from `0.05455` at step 0 to `0.00191` by step 600, GPU memory remained
  `7552 MiB / 24570 MiB`, and temperature was `75C`. No OOM or resource contention observed.
  T2.3 tier 3 acceptance remains pending until training and fidelity complete.
- 2026-07-06T08:47Z: WM tier 3 training and fidelity completed successfully. Fidelity wrote
  tier 3 rows to `results/wm_fidelity.parquet`: horizon 1 PSNR `30.276`, SSIM `0.984`,
  LPIPS `0.046`; horizon 8 PSNR `26.420`, SSIM `0.955`, LPIPS `0.079`; horizon 16 PSNR
  `24.948`, SSIM `0.938`, LPIPS `0.096`; horizon 32 PSNR `22.541`, SSIM `0.900`,
  LPIPS `0.121`; mean divergence step `20.375`.
- 2026-07-06T14:19Z: Started WM tier 4 concurrently with the clean T2.2 rerun in tmux
  session `t23_wm_tier4` after confirming tier 3 had finished and GPU memory was low
  (`1357 MiB / 24570 MiB`). Command sequence mirrors tiers 2-3: `scripts/train_wm_tier.sh 4`,
  then tier 4 fidelity if training exits successfully. W&B run: `y3ueubkp` (`wm_tier4`).
  Startup check: T2.2 and T2.3 both alive; tier 4 loaded 126,546 clips, VAE loss decreased
  from `0.05597` at step 0 to `0.00301` by step 400, GPU memory remained
  `7262 MiB / 24570 MiB`, and temperature was `71C`. No OOM or resource contention observed.
  T2.3 tier 4 acceptance remains pending until training and fidelity complete.
- 2026-07-07T00:44Z: WM tier 4 training and fidelity completed successfully. Fidelity wrote
  tier 4 rows to `results/wm_fidelity.parquet`: horizon 1 PSNR `31.981`, SSIM `0.990`,
  LPIPS `0.030`; horizon 8 PSNR `29.318`, SSIM `0.977`, LPIPS `0.040`; horizon 16 PSNR
  `27.487`, SSIM `0.965`, LPIPS `0.045`; horizon 32 PSNR `24.998`, SSIM `0.946`,
  LPIPS `0.056`; mean divergence step `23.875`.
- 2026-07-08T05:03Z: Started WM tier 5 (user-approved) concurrently with T2.4 in tmux
  session `t23_wm_tier5`. Tier 5 had never been started before this — tiers 3 and 4
  completed during the T2.2 rerun period, but no tier 5 launch existed until now. Command
  sequence mirrors tiers 2-4: `scripts/train_wm_tier.sh 5`, then tier 5 fidelity only if
  training exits successfully, logging to `logs/t2.3_wm_tier5.log`. W&B run: `a6i1aadp`
  (`wm_tier5`). Startup check: tier 5 loaded 122,706 clips (clip_len 9), VAE step 0 loss
  `0.05535`; T2.4 and T2.3 both alive; GPU memory `10935 MiB / 24570 MiB`, utilization
  97%, temperature `72C`. No OOM. Because tier 5 adds an LPIPS loss term on top of latent
  MSE, do not extrapolate runtime from tiers 3-4 alone; a ~1h throughput check-in is
  scheduled to produce a measured completion estimate. T2.3 tier 5 acceptance remains
  pending until training and fidelity complete.
