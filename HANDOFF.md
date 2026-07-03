# Type 1 -> Type 2 Handoff

Status date: 2026-07-02.

Type 1 is now verified from a clean clone of the pushed GitHub repository.

- GitHub repository: https://github.com/ZaidGhazal/world-models-eval
- Release tag: `v0.1-type1-complete`
- Validated code commit: `9b5b545dfdb5e41b5cc882dfeb928635a18a69de`
- HF dataset: https://huggingface.co/datasets/zaid9876/world-models-eval
- HF dataset codebase tag: `v3.0` for LeRobot 0.4.4
- Public repo access: the GPU machine can clone over HTTPS or SSH. No deploy key is required.

## What Was Built

- **Data pipeline** (`dreamgrasp/data/`): LIBERO spatial/object/goal HDF5s converted into a
  LeRobotDataset v3 artifact: 1500 episodes, 200,485 frames, about 600 MB, AV1 video.
  Action normalization uses train-split stats in `configs/norm_stats.json`. Splits are frozen
  in `configs/splits.json`: 960 train / 120 val / 120 test / 300 heldout.
- **Policy stack** (`dreamgrasp/policy/train.py`): SmolVLA fine-tuning through LeRobot 0.4.4
  factory/processor APIs, OmegaConf configs, W&B logging, and checkpoint save/reload.
- **Simulator eval** (`dreamgrasp/eval/sim_eval.py`): seeded LIBERO rollouts, success predicate,
  MP4 capture, Wilson CIs, and `results/sim_success.parquet`.
- **World model** (`dreamgrasp/world_model/`): custom FrameVAE plus block-causal dynamics
  transformer with a proprio state head, tier configs 1-5 plus tiny, and fidelity metrics.
- **Dream eval** (`dreamgrasp/eval/dream_eval.py`): policy acts on decoded dreamed frames;
  proprio comes from the dynamics state head.
- **Success classifier** (`dreamgrasp/eval/success_classifier.py`): frozen SigLIP-base features
  with temporal pooling and MLP head.
- **Analysis and scaffolds**: correlation analysis, notebook, report skeleton, Gradio Space
  scaffold, CPU CI config, and scripts.

## Clean-Clone Verification

Verification clone:
`/tmp/world-models-eval-clean.7TidV3/world-models-eval`

Clone command used:

```bash
git clone --branch v0.1-type1-complete git@github.com:ZaidGhazal/world-models-eval.git /tmp/world-models-eval-clean.7TidV3/world-models-eval
```

The original clean clone checked out commit `b623c8a3a57901f5cb2e1ef9ee3aef7f1a5406ad`. GPU
bring-up later exposed two fresh-install dependency gaps; the release tag now points at
`9b5b545dfdb5e41b5cc882dfeb928635a18a69de`, which adds:

- `gradio==4.44.1` instead of `gradio==6.19.0`, because Gradio 6.19 requires
  `huggingface_hub>=1.2.0` while LeRobot 0.4.4 is pinned to the 0.35.x line.
- `future==1.0.0`, because `bddl==1.0.1` imports `future.utils` at runtime.

| Check | Clean-clone result |
|---|---|
| Install package | PASS: `python -m pip install -e . --no-deps` |
| LIBERO setup | PASS: `scripts/setup_libero.sh`, pinned commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| HF dataset download | PASS: `snapshot_download(... revision="v3.0")` into `data/lerobot/world-models-eval` |
| LeRobot local load | PASS: episode 0 loads, 138 frames, dataset metadata codebase `v3.0` |
| Smoke test | PASS: LIBERO render, SmolVLA forward on MPS, W&B logging |
| Full test suite | PASS: `python -m pytest` -> 19 passed |
| Lint | PASS: `ruff check .` |
| Types | PASS: `mypy dreamgrasp` |
| Policy tiny train | PASS: 200 steps, loss first10 `0.3380` -> last10 `0.0465`; checkpoint saved |
| Policy overfit-one-batch | PASS: 200 steps, loss first10 `0.2935` -> last10 `0.0119` |
| Simulator eval tiny | PASS: 2 tasks x 3 rollouts, 6 parquet rows, videos written |
| World-model tiny train | PASS: VAE loss `0.03084` -> `0.00750`; dynamics loss `1.17244` -> `0.01521` |
| Fidelity module | PASS: horizons 1 and 8 wrote PSNR/SSIM/LPIPS/divergence rows |
| Dream loop tiny | PASS: 2 dreams x 20 frames, parquet and videos written |
| Classifier training loop | PASS: 6 labeled tiny videos, 1 epoch, checkpoint head saved |
| Synthetic correlation | PASS: target 0.95 -> 0.988; target 0.0 -> -0.014 |
| No CUDA literals | PASS via `tests/test_no_cuda_literals.py` |
| Tier configs | PASS: `tier_1.yaml` through `tier_5.yaml` and `tiny.yaml` present |
| macOS notes | PASS: `docs/macos.md` present |
| CI config | PASS: `.github/workflows/ci.yml` present |

The generated clean-clone result schemas were:

- `results/sim_success.parquet`: `(6, 5)`, columns
  `checkpoint, task, seed, success, steps`.
- `results/wm_fidelity.parquet`: `(2, 7)`, columns
  `checkpoint, split, horizon, psnr, ssim, lpips, divergence_step`.
- `results/dream_success.parquet`: `(2, 5)`, columns
  `checkpoint, task, wm_tier, seed, dream_success_prob`.

## GPU Bring-Up Verification

GPU machine:

- SSH target: `umd-user@141.215.80.58`
- Hostname: `umd-004061`
- OS: Ubuntu 22.04, Linux 6.8
- GPU: NVIDIA RTX 4500 Ada Generation, 24,570 MiB VRAM
- Driver: 535.309.01; `nvidia-smi` reports CUDA 12.2
- Project checkout: `~/world-models-eval`
- Conda env: `world-models-eval`

GPU setup completed:

- `sudo apt-get install -y ffmpeg libegl1 libgl1 libosmesa6-dev`
- `git checkout v0.1-type1-complete`; GPU validation ran on code commit
  `9b5b545dfdb5e41b5cc882dfeb928635a18a69de`
- `python -m pip install -e ".[dev]"` PASS
- `python -m pip check` PASS
- `scripts/setup_libero.sh` PASS, pinned LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- HF dataset snapshot downloaded at revision `v3.0`
- W&B login configured from `w&b.secret`

GPU parity checks with `MUJOCO_GL=egl`:

| Check | GPU result |
|---|---|
| CUDA availability | PASS: PyTorch `2.10.0+cu126`, CUDA true, RTX 4500 Ada |
| LeRobot dataset load | PASS: episode 0 loads, 138 frames, metadata codebase `v3.0` |
| Full test suite | PASS: `python -m pytest` -> 19 passed |
| Lint | PASS: `ruff check .` |
| Types | PASS: `mypy dreamgrasp` |
| Compile/import | PASS: `compileall`; `space.app.build()` returns `Blocks` |
| Smoke test | PASS: LIBERO EGL render, SmolVLA forward on CUDA, W&B logging |
| Synthetic correlation | PASS: target 0.95 -> 0.988; target 0.0 -> -0.014 |
| Policy tiny train | PASS: 200 steps, loss first10 `0.3397` -> last10 `0.0465`; checkpoint saved |
| Policy overfit-one-batch | PASS: 200 steps, loss first10 `0.2899` -> last10 `0.0129` |
| Simulator eval tiny | PASS: 2 tasks x 3 rollouts, 6 parquet rows, videos written |
| World-model tiny train | PASS: VAE loss `0.03192` -> `0.00768`; dynamics loss `1.11056` -> `0.01465` |
| Fidelity module | PASS: horizons 1 and 8 wrote PSNR/SSIM/LPIPS/divergence rows |
| Dream loop tiny | PASS: 2 dreams x 20 frames, parquet and videos written |
| Classifier training loop | PASS: 6 labeled tiny videos, 1 epoch, checkpoint head saved |

GPU result schemas:

- `results/sim_success.parquet`: `(6, 5)`, columns
  `checkpoint, task, seed, success, steps`.
- `results/wm_fidelity.parquet`: `(2, 7)`, columns
  `checkpoint, split, horizon, psnr, ssim, lpips, divergence_step`.
- `results/dream_success.parquet`: `(2, 5)`, columns
  `checkpoint, task, wm_tier, seed, dream_success_prob`.

## HF Dataset Status

- Dataset repo: `zaid9876/world-models-eval`
- Codebase tag: `v3.0`, target commit `d56cbc3c472efc52895be7a9ad4b35f7612db3c5`
- Dataset-server checks:
  - `/splits`: 200, no `CastError`, no `StreamingRowsError`
  - `/first-rows`: 200, no `CastError`, no `StreamingRowsError`
  - `/is-valid`: `viewer=true`, `preview=true`, `filter=true`, `statistics=true`
  - `/parquet`: 200 with generated train parquet
  - `/size`: 200 with 200,485 rows
- The public Hub page rendered with the viewer after a no-cache request. A stale bare-page cache
  previously still contained the old CastError, but the live backend and no-cache page are fixed.

## GitHub And GPU Access

The repository is public. On the GPU machine:

```bash
git clone https://github.com/ZaidGhazal/world-models-eval.git
cd world-models-eval
git checkout v0.1-type1-complete
```

SSH is also fine:

```bash
git clone git@github.com:ZaidGhazal/world-models-eval.git
cd world-models-eval
git checkout v0.1-type1-complete
```

The read-only deploy key from the request is not needed while the repository remains public.

## Deviations From The Guide

1. **Images at native 128x128, not 256x256**: LIBERO raw demos are 128px. Upscaling would
   fabricate pixels and increase storage.
2. **mujoco 2.3.7 / robosuite 1.4.1**: newer versions broke LIBERO/robosuite compatibility.
3. **Custom small VAE instead of frozen SD-VAE**: rationale is in `world_model/vae.py`.
4. **LeRobot 0.4.4 API**: policies use processor-pipeline batches.
5. **LIBERO quirks handled in code**: legacy editable install, local LIBERO config, pickle
   compatibility, and OpenGL frame flipping.

## Undocumented Scope Audit

The previous uncommitted `IMPLEMENTATION_GUIDE.md` contained unsupported uncertainty-estimation
and counterfactual-diagnostics sections (`T2.3.5` and `T2.5.5`). They were not in the committed
guide, no `RUNBOOK.md` exists in this workspace, and there is no code/test/script/doc dependency
on those tasks. They were removed as erroneous scope.

If you want to add them later, treat them as new Type 2+ scope:

- Uncertainty estimation would produce per-tier uncertainty summaries and analyze whether they
  predict dream/sim disagreement.
- Counterfactual diagnostics would search for minimal dreamed action changes that flip model-
  predicted failures to successes, producing model-based explanations only.

## Known Type 2 Notes

- Policy action chunk is 8 while SmolVLA base pretraining uses chunk 50; if fine-tuning underperforms,
  try chunk 50 first.
- Delete Type 1 smoke `results/*.parquet` before real Type 2 runs.
- Success-classifier accuracy is meaningless until real T2.2 videos contain both classes.
- Dream-loop wall-clock on MPS was slow; CUDA budget must be checked in Type 2.
