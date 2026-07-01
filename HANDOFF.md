# Type 1 → Type 2 Handoff

Everything below was implemented and verified on a MacBook Pro (M1 Pro, MPS, fp32).
Type 2 starts at tag `v0.1-type1-complete` on a Linux + NVIDIA machine (see §T2.0 of
IMPLEMENTATION_GUIDE.md). Re-run this whole gate there as the parity check.

## What was built

- **Data pipeline** (`dreamgrasp/data/`): one-command conversion of LIBERO
  spatial/object/goal HDF5s → a single LeRobotDataset v3 (1500 episodes, 200,485 frames,
  ~600 MB, AV1 video). Per-dim [-1,1] action normalization from train-split stats
  (`configs/norm_stats.json`), deterministic frozen splits with content-hash leakage guard
  (`configs/splits.json`: 960 train / 120 val / 120 test / 300 heldout, 2 held-out tasks
  per suite). Loaders for policy batches (action chunks) and world-model clips.
- **Policy stack** (`dreamgrasp/policy/train.py`): SmolVLA fine-tuning via lerobot 0.4.4
  factory + processor pipeline; OmegaConf configs; W&B; checkpoint save/reload including
  pre/post-processors. Type 2 config `configs/policy/smolvla_libero.yaml` (bf16 comes from
  device policy, 40k steps, effective bs64).
- **Sim eval** (`dreamgrasp/eval/sim_eval.py`): seeded LIBERO rollouts from per-task init
  states, LIBERO `check_success`, MP4 capture, Wilson 95% CIs, appends to
  `results/sim_success.parquet` `[checkpoint, task, seed, success, steps]`.
- **World model** (`dreamgrasp/world_model/`): FrameVAE (own small VAE, not SD-VAE —
  rationale in `vae.py` docstring) + block-causal dynamics transformer over
  (latent tokens, action token) with a proprio state head. All 5 tier configs + tiny.
  Fidelity: PSNR/SSIM/LPIPS at {1,8,16,32} + divergence step → `results/wm_fidelity.parquet`.
- **Dream eval** (`dreamgrasp/eval/dream_eval.py`): policy acts on decoded dreamed frames;
  proprio from the dynamics state head; wrist camera omitted (SmolVLA masks missing cams);
  seeds from real val clips → `results/dream_success.parquet`.
- **Success classifier** (`dreamgrasp/eval/success_classifier.py`): frozen SigLIP-base,
  mean+max temporal pooling, MLP head; trains from sim-eval videos + labels parquet.
- **Analysis** (`dreamgrasp/eval/correlate.py` + notebook): Spearman ρ with bootstrap CIs,
  task-level Pearson, trust-region chart, synthetic self-check.
- **Scaffolds**: report skeleton, Gradio space (verified serving HTTP 200 locally), CI
  (CPU-only GitHub Actions), scripts/ incl. `reproduce.sh`, `run_study.sh`.

## Verification results (all on MPS, final dataset)

| Check | Result |
|---|---|
| Smoke test (LIBERO render / SmolVLA fwd / W&B) | PASS |
| Data tests + split leakage + norm round-trip | PASS (8 tests) |
| Policy tiny (200 steps, bs2, 5 eps, 64px) | loss 0.338 → 0.047, 170 s |
| Overfit-one-batch (300 steps) | loss 0.297 → **0.009** (< 0.05 bar) |
| Checkpoint reload + inference | PASS (action (1,7)) |
| Sim eval tiny (2 tasks × 3 rollouts) | 0/3 successes each (untrained — expected), parquet + videos OK |
| WM tiny (VAE 150 + dyn 150 steps) | VAE 0.031→0.0075; dyn 1.17→0.015, 19 s |
| 10-step dream rollout decode | blurry-but-plausible frames (see guide: "blurry is fine") |
| Fidelity module (val) | PSNR 18.6 / SSIM 0.78 / LPIPS 0.55 @ h=1; divergence 8.0 |
| Dream loop e2e (20 steps + classifier) | probs ~0.002–0.004, valid parquet |
| Classifier training loop (56 videos) | runs; val_acc trivial (all-failure labels at tiny scale) |
| Synthetic correlation recovery | target 0.95 → 0.988; target 0.0 → −0.014 |
| pytest / ruff / mypy | all green |
| Loader throughput (M1, workers=0) | policy 425 f/s; WM clips 825 f/s |

## Deviations from the guide (details in docs/macos.md, docs/datasets.md)

1. **Images at native 128×128, not 256×256** — LIBERO raw demos are 128px; upscaling
   fabricates pixels. WM spec (128px) unaffected; SmolVLA upscales internally.
2. **mujoco 2.3.7 / robosuite 1.4.1** — robosuite 1.5 broke LIBERO; mujoco 3.x broke
   robosuite 1.4 (`mj_fullM` signature).
3. **Own small VAE instead of frozen SD-VAE** — rationale in `world_model/vae.py`.
4. **lerobot 0.4.4 API** — policies consume processor-pipeline batches
   (`make_pre_post_processors`), not raw dicts as in the guide's era.
5. **LIBERO quirks handled in our code, not by patching third_party**: legacy editable
   install, pre-seeded `~/.libero/config.yaml`, `weights_only=False` for init-state pickles,
   opengl (bottom-up) frame convention flipped at capture.

## Known issues for Type 2

- **Policy action chunk is 8** (guide) vs. smolvla_base pretraining chunk 50 — loads fine
  (per-step projections), but if fine-tuning underperforms, try chunk 50 first.
- One transient crash was observed during a bulk conversion while three MPS jobs ran
  concurrently (memory pressure); a clean re-run completed. Don't convert while training.
- `results/*.parquet` from Type 1 are tiny-scale smoke artifacts — delete before real runs.
- Success-classifier val accuracy is meaningless until T2.2 produces both classes; the
  ≥90% held-out bar applies there.
- Dream-loop wall-clock on MPS ≈ 1.5 s/dreamed step at 64px (dominated by SmolVLA
  select_action); budget check (<10 min per triple at N=50, T=200) must be done on CUDA.

## Blocked/external items

- **HF dataset publish** to `world-model-eval/dreamgrasp-libero`: ready
  (`python -m dreamgrasp.data.convert_libero` output + card); requires `hf auth login`
  on this machine. Status at handoff: (see final commit message / README badge).
