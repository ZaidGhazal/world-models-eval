# Type 1 → Type 2 Handoff

Status date: 2026-07-02.

This repository has real local git history on `main`, but it has not yet been pushed to GitHub
because no GitHub remote exists and this machine has SSH auth but no repo-creation CLI/API token.
Do not start Type 2 until the GitHub repo is created, this history is pushed, the
`v0.1-type1-complete` tag is pushed, and the T1.7 gate is re-run from a clean clone.

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

## Current Verification

The full historical Type 1 gate was reported in the prior local commits, but it has not yet been
re-verified from a clean clone of a pushed GitHub repo. Treat the table below as local evidence,
not final Type 1 release evidence.

| Check | Status |
|---|---|
| Local git history | PRESENT: 9 commits on `main`; no remote; no tag |
| HF dataset | PUBLIC: `zaid9876/world-models-eval` |
| HF dataset viewer | FIXED at dataset-server `/first-rows`; page cache may lag |
| LeRobot codebase tag needed | `v3.0` from installed LeRobot 0.4.4; not created yet |
| Focused local tests after rename | PASS: `python -m pytest tests/test_data.py tests/test_norm.py tests/test_splits.py tests/test_shapes.py tests/test_correlate.py tests/test_no_cuda_literals.py` → 19 passed |
| Local lint/type checks after rename | PASS: `ruff check .`; `mypy dreamgrasp` |
| Clean-clone T1.7 gate | NOT RUN; blocked until GitHub repo is pushed |

## Prior Type 1 Evidence To Re-Run From Clean Clone

These results were recorded before the GitHub-push gap was found and must be re-run from the
fresh clone before declaring `v0.1-type1-complete`:

| Check | Prior local result |
|---|---|
| Smoke test (LIBERO render / SmolVLA forward / W&B) | PASS |
| Data tests + split leakage + norm round-trip | PASS |
| Policy tiny (200 steps, bs2, 5 eps, 64px) | loss 0.338 → 0.047 |
| Overfit-one-batch (300 steps) | loss 0.297 → 0.009 |
| Checkpoint reload + inference | PASS |
| Sim eval tiny (2 tasks × 3 rollouts) | parquet + videos OK |
| WM tiny (VAE 150 + dyn 150 steps) | VAE and dynamics losses decreased |
| 10-step dream rollout decode | produced plausible blurry frames |
| Fidelity module (val) | produced PSNR/SSIM/LPIPS/divergence metrics |
| Dream loop e2e (20 steps + classifier) | wrote valid parquet |
| Classifier training loop (56 videos) | ran; accuracy meaningless at tiny scale |
| Synthetic correlation recovery | target 0.95 → 0.988; target 0.0 → -0.014 |
| Loader throughput (M1, workers=0) | policy ~425 f/s; WM clips ~825 f/s |

## Deviations From The Guide

1. **Images at native 128×128, not 256×256**: LIBERO raw demos are 128px. Upscaling would
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

## Blocking Items Before Type 2

1. **GitHub repo creation/push is still missing.**
   - Local repo exists with real history.
   - No remote exists.
   - SSH auth works as GitHub user `ZaidGhazal`.
   - `gh`/`hub` are not installed and no GitHub API token is available in the shell.
   - Once the repo exists, push `main`, create and push `v0.1-type1-complete`, then clone it into
     a temp directory and run the full T1.7 gate there.
2. **HF codebase-version tag still needs approval.**
   - Installed LeRobot 0.4.4 reports `CODEBASE_VERSION = "v3.0"`.
   - Required command to approve: `HfApi().create_tag(repo_id="zaid9876/world-models-eval", repo_type="dataset", tag="v3.0", token=token)`.
3. **GPU access instructions depend on the final GitHub repo visibility.**
   - If public: GPU machine can clone the HTTPS/SSH URL.
   - If private: add this read-only deploy key to the GitHub repo:
     `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEE1LSgx6k64mOwuTb12eBzLjrjjge0J1TO2HGryOH3V`.

## Known Type 2 Notes

- Policy action chunk is 8 while SmolVLA base pretraining uses chunk 50; if fine-tuning underperforms,
  try chunk 50 first.
- Delete Type 1 smoke `results/*.parquet` before real Type 2 runs.
- Success-classifier accuracy is meaningless until real T2.2 videos contain both classes.
- Dream-loop wall-clock on MPS was slow; CUDA budget must be checked in Type 2.
