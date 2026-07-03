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
