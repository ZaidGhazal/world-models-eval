# Simulator Failure Gallery

T2.2 fills this file from real simulator evaluation videos. Do not use tiny-run videos here.

## Selection Rule

After `results/sim_success.parquet` and `results/videos/*.mp4` are produced, select 20 failures
that cover different checkpoints, tasks, and seeds. Prefer failures that explain why the worst
and best policy checkpoints differ, because the Type 2 rank-correlation study requires at least
a 25-point spread.

## Categories

Use these categories unless the real failures suggest clearer labels:

- Object miss or poor approach
- Grasp failure
- Collision or workspace violation
- Wrong object or wrong target
- Partial completion
- Timeout after slow progress
- Recovery failure after contact
- Camera/state mismatch suspicion

## Real Failure Entries

| # | Checkpoint | Task | Seed | Category | Video | Notes |
|---:|---|---|---:|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD | Fill after T2.2 |
