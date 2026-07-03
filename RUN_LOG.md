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
