# Type 2 Readiness Record

Status date: 2026-07-02.

This record covers the preparation-only Type 2 session. No full-scale multi-hour training or
evaluation runs were launched.

## Machine

- SSH target: `umd-user@141.215.80.58`
- Hostname: `umd-004061`
- GPU: NVIDIA RTX 4500 Ada Generation
- VRAM: 24,570 MiB
- Driver: 535.309.01
- `nvidia-smi` CUDA: 12.2
- PyTorch: `2.10.0+cu126`
- Torch CUDA runtime: 12.6
- LeRobot: 0.4.4
- Conda env: `world-models-eval`
- MuJoCo backend: `MUJOCO_GL=egl`

## Environment And Dataset

- `python -m pip install -e ".[dev]"`: PASS
- `python -m pip check`: PASS
- CUDA visible to torch on the RTX 4500 Ada: PASS
- LIBERO setup via `scripts/setup_libero.sh`: PASS at commit
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Dataset source recorded in configs:
  `https://huggingface.co/datasets/zaid9876/world-models-eval`, revision `v3.0`
- Dataset local load: PASS, episode 0 loads with 138 frames and codebase tag `v3.0`
- `configs/splits.json` matches HF dataset snapshot:
  `fbb5d4e9cf238e27303bd30f3b4f95c07faf38dfd42bdcc17c24c690f09032bc`
- `configs/norm_stats.json` matches HF dataset snapshot:
  `65510e254754d338121312e1a0395aa984a48a40c618a9f306fb68aa5a7a8fda`

## Parity Gate On CUDA

| Check | Result |
|---|---|
| Smoke test | PASS: LIBERO EGL render, SmolVLA CUDA forward, W&B logging |
| Full test suite | PASS: `python -m pytest` -> 19 passed |
| Lint | PASS: `ruff check .` |
| Types | PASS: `mypy dreamgrasp` |
| Compile/import | PASS: `compileall`; `space.app.build()` returns `Blocks` |
| Policy tiny train | PASS |
| Policy overfit-one-batch | PASS |
| Simulator eval tiny | PASS |
| World-model tiny train | PASS |
| Fidelity module | PASS |
| Dream loop tiny | PASS |
| Classifier training loop | PASS |
| Synthetic correlation | PASS: 0.95 -> 0.988; 0.0 -> -0.014 |

## Type 2 Dry-Run Launches

All dry-runs were run on the real GPU with `MUJOCO_GL=egl`.

| Command | Result |
|---|---|
| `scripts/train_policy.sh --dry-run` | PASS: 20 steps, checkpoint `checkpoints/dry_run/policy/step_000020` |
| `scripts/train_wm_tier.sh 1 --dry-run` | PASS: VAE/dynamics checkpoints written |
| `scripts/train_wm_tier.sh 2 --dry-run` | PASS: VAE/dynamics checkpoints written |
| `scripts/train_wm_tier.sh 3 --dry-run` | PASS: 12L/512d dynamics path written |
| `scripts/train_wm_tier.sh 4 --dry-run` | PASS: 12L/512d dynamics path written |
| `scripts/train_wm_tier.sh 5 --dry-run` | PASS: context-8 + LPIPS/VGG path written |
| `scripts/sim_eval.sh --dry-run` | PASS: 1 rollout row + video |
| `scripts/dream_eval.sh --dry-run` | PASS: 1 dream row + video |
| `scripts/run_study.sh --dry-run` | PASS: synthetic correlation recovery |

Dry-run schemas:

- `results/dry_run/sim_success.parquet`: `(1, 5)`, columns
  `checkpoint, task, seed, success, steps`.
- `results/dry_run/dream_success.parquet`: `(1, 5)`, columns
  `checkpoint, task, wm_tier, seed, dream_success_prob`.

## Credentials

- W&B: PASS via real synced smoke run to project `world-models-eval`.
- HF write token: PASS via temporary private dataset repo create, 1 KB upload,
  file delete, and repo delete. The production dataset was not modified.

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

## Status

Ready for Prompt 2B; do not start full-scale training until explicitly approved.
