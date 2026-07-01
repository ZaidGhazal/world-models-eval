# DreamGrasp — Developer Implementation Guide (A → Z)

This is the complete build spec for DreamGrasp: a single-GPU, fully reproducible study of **how good a learned world model must be before you can trust it to evaluate robot policies**.

> **Positioning rule (non-negotiable):** This project *builds on* WorldEval (arXiv:2505.19017), WPE (arXiv:2506.00613), Ctrl-World (arXiv:2510.10125), and SIMPLER (arXiv:2405.05941). We never claim to have invented world-model-based evaluation. Our contributions are (1) the **quality→reliability calibration curve** and (2) the **open single-GPU harness**. Cite these papers in the README, the report, and the model cards.

> **Version caveat:** Library APIs (LeRobot, LIBERO, diffusers) change fast. Pin every version in `pyproject.toml` on day 1, and verify each install command against current docs before running.

---

## Two-machine execution model — READ THIS FIRST

The project is executed in **two sequential work types on two different machines**:

| | **TYPE 1 — Development** | **TYPE 2 — Training & Study** |
|---|---|---|
| Machine | **MacBook Pro, M1 Pro chip** (Apple Silicon, unified memory, no CUDA) | Linux box / cloud instance with **NVIDIA GPU ≥24 GB VRAM** (RTX 4090 or A100) |
| Compute backend | CPU + PyTorch **MPS** | **CUDA** (bf16) |
| Scope | All code, all tests, full data pipeline, tiny-scale sanity training, mocked end-to-end runs | Real training runs, bulk simulator/dream rollouts, the calibration study, releases |
| Exit condition | The **Type 1 Handoff Gate** (§T1.7) passes entirely on the Mac | Definition of Done (§T2.8) |

**Workflow contract for the development agent:** implement and verify everything in Type 1 on the Mac first. Only after every Handoff Gate item passes does the codebase move (via git) to the GPU machine, where a second instruction begins Type 2. **No Type 2 task may be started on the Mac**, and no Type 1 code should need rewriting on the GPU machine — only config changes.

### Rules that make the two-machine split work (enforced from the first commit)

1. **Device-agnostic code everywhere.** One utility, used by every script:
   ```python
   # dreamgrasp/utils/device.py
   def get_device():
       if torch.cuda.is_available(): return "cuda"
       if torch.backends.mps.is_available(): return "mps"
       return "cpu"
   ```
   Never hard-code `"cuda"`. Precision policy: bf16 on CUDA, fp32 on MPS/CPU (MPS half-precision support is unreliable — do not fight it).
2. **No CUDA-only dependencies in the core package.** flash-attention, bitsandbytes, TensorRT, xformers are forbidden in `dreamgrasp/` (optional CUDA extras may live behind a `[cuda]` extra and a runtime import guard). OpenVLA-7B + LoRA is a Type 2 *optional* branch for this reason — the default policy is SmolVLA.
3. **Rendering backend is configurable.** MuJoCo headless rendering: `MUJOCO_GL=egl` on Linux, default GLFW/CGL on macOS. Read it from env; never hard-code.
4. **Every training script has a `--tiny` mode** (≤5 episodes, 64px, ≤200 steps) that must run to completion on the Mac. Tiny mode is how Type 1 proves the code works without GPU-scale compute.
5. **Configs, not code, define scale.** The difference between a Mac sanity run and a real A100 run must be exactly one config file.

---

# TYPE 1 — Development on the MacBook Pro (M1 Pro)

Everything in this section runs and is verified on the Mac. There is no time schedule — the agent executes tasks T1.0 → T1.7 in order, and each task is complete when its acceptance criteria pass, not when a date arrives.

## T1.0 Environment setup (macOS / Apple Silicon)

**Hardware assumptions:** M1 Pro, 16–32 GB unified memory, ≥200 GB free disk (full raw dataset can stay on an external drive or be downloaded partially).

```bash
# Homebrew deps
brew install ffmpeg git-lfs

# Project env (arm64 native — do NOT use x86 conda under Rosetta)
conda create -n dreamgrasp python=3.10 -y && conda activate dreamgrasp
pip install torch torchvision            # official arm64 wheels include MPS

# Core deps
pip install lerobot
pip install mujoco robosuite             # MuJoCo ships official Apple Silicon wheels
git clone https://github.com/Lifelong-Robot-Learning/LIBERO && pip install -e ./LIBERO
pip install wandb hydra-core einops webdataset huggingface_hub gradio plotly
pip install ruff mypy pytest
```

**Known macOS gotchas (document any new ones in `docs/macos.md`):**
- Do **not** set `MUJOCO_GL=egl` — that's Linux. On macOS, offscreen rendering works with the default backend; if a robosuite version insists on EGL, patch the renderer selection, don't switch machines.
- `torch.compile` is unreliable on MPS — guard it: enable only when device is cuda.
- If unified memory pressure kills a run, drop `--tiny` resolution to 64px; never raise swap.

**Smoke test (must pass before anything else):**
1. Render one LIBERO episode offscreen on the Mac and save an MP4.
2. Load `lerobot/smolvla_base` and run one forward pass on MPS.
3. `wandb login` and log a dummy metric.

Create the repo skeleton (§Repository layout) and commit: env files, smoke test, README stub, `utils/device.py`.

## T1.1 Data pipeline (full implementation — this is real Type 1 work, not a mock)

The pipeline is CPU-bound, so the Mac does the *actual* work here; the processed dataset it produces is the one used in Type 2.

1. **Download** LIBERO demo HDF5s for `libero_spatial`, `libero_object`, `libero_goal` (add `libero_10` later if compute allows).
2. **Convert** to LeRobotDataset format via `dreamgrasp/data/convert_libero.py`:

| Field | Spec |
|---|---|
| `observation.images.agentview` | 256×256 RGB, MP4-encoded |
| `observation.images.wrist` | 256×256 RGB |
| `observation.state` | float32 proprio vector |
| `action` | 7-DoF delta EEF + gripper, **normalized to [-1,1] per-dim from training-set stats** |
| `task` | language instruction string |
| `episode_index`, `frame_index`, `timestamp` | standard LeRobot fields |

3. **Normalization stats** → `configs/norm_stats.json`, shared by policy and world models. Unit-test the round trip `normalize(denormalize(x)) == x` — this is the most common silent bug in the whole project.
4. **Splits, fixed forever:** 80% train / 10% val per task, plus **2–3 whole held-out tasks per suite** excluded from all training (they test evaluation under distribution shift). Add a leakage unit test (no held-out episode hash in train).
5. **Loaders** (`dreamgrasp/data/loader.py`): batch modes for (a) policy training and (b) world-model training. Benchmark throughput on the Mac and record it in `docs/datasets.md` (M1 numbers are fine; re-benchmark in Type 2).
6. **Publish** the processed dataset + dataset card to HF Hub (`your-handle/dreamgrasp-libero`) — can be done from the Mac. Note LIBERO's MIT license and cite the benchmark.

**T1.1 acceptance:** one command converts raw → LeRobotDataset; all data unit tests green; dataset live on HF.

## T1.2 Policy code (implementation + tiny verification)

Write the complete SmolVLA fine-tuning stack — config, trainer, checkpointing, W&B logging, augmentation — exactly as it will run in Type 2:

```
configs/policy/smolvla_libero.yaml      ← real Type 2 config (bf16, 40k steps, bs64)
configs/policy/smolvla_tiny.yaml        ← Mac config (fp32, 200 steps, bs2, 64px, 5 episodes)
```

**Mac verification (required):**
- `python -m dreamgrasp.policy.train --config configs/policy/smolvla_tiny.yaml` completes on MPS, loss decreases, a checkpoint is saved and reloadable.
- Overfit-one-batch test: loss → ~0 on a single batch within 300 steps (proves the learning loop is wired correctly).

## T1.3 Simulator evaluation harness (implementation + tiny verification)

Implement `dreamgrasp/eval/sim_eval.py` fully: seeded rollouts, LIBERO success predicate, video capture every 10th rollout, Wilson 95% CIs, `results/sim_success.parquet` schema `[checkpoint, task, seed, success, steps]`.

**Mac verification:** run N=3 rollouts of the (untrained) tiny checkpoint on 2 tasks. Success will be ~0% — that's fine; what must work is the loop, the seeding, the videos, and the parquet output.

## T1.4 World-model code (implementation + tiny verification)

Implement the full architecture and the tier system:
- **Frame autoencoder:** VAE compressing 128×128 → 16×16 latents (or a frozen pretrained SD-VAE from `diffusers` — decide now, document why).
- **Dynamics transformer:** causal transformer over (frame-latent tokens + action tokens); base ~12L/512d (~50–80M params); loss = latent MSE (+LPIPS option for the top tier).
- **All five tier configs** written now, even though only Type 2 trains them:

| Tier | Data | Context | Dynamics |
|---|---|---|---|
| WM-1 | 10% | 2 frames | small (6L/256d) |
| WM-2 | 25% | 2 | small |
| WM-3 | 50% | 4 | base |
| WM-4 | 100% | 4 | base |
| WM-5 | 100% | 8 | base + LPIPS |

- **Fidelity metrics module** (`world_model/fidelity.py`): PSNR/SSIM/LPIPS at horizons {1,8,16,32} + rollout-divergence step.

**Mac verification:** `--tiny` world-model training (64px, 5 episodes, small dynamics) completes on MPS with decreasing loss; a 10-step rollout decodes to visually plausible (blurry is fine) frames; fidelity module returns numbers on the tiny val split.

## T1.5 Dream-rollout loop + success classifier (implementation + mocked verification)

- Implement `eval/dream_eval.py`: policy acts on **decoded dreamed frames**; proprio inside the dream from a small state head on the dynamics model (or open-loop integration — document the choice); T=200 dreamed steps, N configurable; output schema `[checkpoint, task, wm_tier, seed, dream_success_prob]`.
- Implement `eval/success_classifier.py`: frozen image encoder (e.g., SigLIP) per-frame + temporal pooling + MLP head, trained on labeled simulator videos.

**Mac verification:** run the full dream loop with the tiny policy + tiny world model for 20 dreamed steps — it must execute end-to-end and write a valid parquet. Train the classifier on ~50 tiny sim videos to prove the training loop runs (accuracy is meaningless at this scale; the ≥90% bar applies in Type 2).

## T1.6 Analysis, report skeleton, and demo scaffold

- `eval/correlate.py` + `notebooks/analysis.ipynb`: Spearman ranking reliability with bootstrap CIs, task-level Pearson, in-distribution vs. held-out split, and the trust-region plotting function. **Verify on synthetic data:** generate fake sim/dream results with a known correlation and confirm the pipeline recovers it.
- `report/report.md` skeleton with the section structure and citation stubs.
- `space/app.py` Gradio scaffold running locally on the Mac with placeholder videos.

## T1.7 TYPE 1 HANDOFF GATE — all boxes must be checked on the Mac

- [ ] Smoke test passes (LIBERO render, SmolVLA forward on MPS, W&B).
- [ ] Full data pipeline done for real; dataset on HF; all data tests green.
- [ ] `pytest` fully green; `ruff` and `mypy` clean; CI configured (GitHub Actions, CPU-only jobs).
- [ ] Policy tiny-train completes on MPS; overfit-one-batch passes.
- [ ] Sim-eval harness produces valid parquet + videos on a tiny run.
- [ ] World-model tiny-train completes; 10-step rollout decodes; all 5 tier configs exist.
- [ ] Dream loop runs end-to-end tiny; classifier training loop runs.
- [ ] Correlation pipeline recovers a known correlation from synthetic data.
- [ ] No `"cuda"` literals outside `utils/device.py` (add a grep test for this).
- [ ] `docs/macos.md` records every Mac-specific workaround.
- [ ] Everything committed and pushed; tag `v0.1-type1-complete`.

**Handoff procedure:** push to GitHub → provision the GPU machine → clone → the processed dataset re-downloads from the HF Hub (no manual file copying) → begin Type 2.

---

# TYPE 2 — Training & study on the GPU machine

Prerequisite: tag `v0.1-type1-complete` checked out on a Linux machine with an NVIDIA GPU (≥24 GB VRAM; A100 40GB preferred). Type 2 is mostly *running* what Type 1 built, plus analysis and release. There is no calendar schedule — tasks T2.0 → T2.8 run in order, each gated by its acceptance criteria. Note that Type 2 durations are bound by GPU wall-clock (training and rollout runs take real hours regardless of agent speed — see the GPU budget table), so the agent should launch long runs, monitor via W&B, and proceed to preparable work (report skeleton, Space assets) while they execute.

## T2.0 Environment bring-up (Linux / CUDA)

```bash
sudo apt-get install -y ffmpeg libegl1 libgl1 libosmesa6-dev
conda create -n dreamgrasp python=3.10 -y && conda activate dreamgrasp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"                  # same pinned deps as the Mac
export MUJOCO_GL=egl                     # Linux headless rendering
```

**Parity check (required before any training):** re-run the entire Type 1 Handoff Gate suite on this machine — smoke test, `pytest`, one tiny policy run, one tiny WM run, tiny dream loop. Everything must pass on CUDA with zero code changes. If anything fails, fix it as a bug (the device abstraction leaked), not with a machine-specific patch.

## T2.1 Policy training (real scale)

- Run `configs/policy/smolvla_libero.yaml`: bf16, bs64 (grad-accum as needed), lr 1e-4 cosine, action chunk 8, ~40k steps on the merged suite. Public W&B run.
- **Save checkpoints at 5k/10k/20k/40k steps + one deliberately early "bad" one** → the rankable set {P1…P5}.
- Optional branch (A100 only): OpenVLA-7B + LoRA (r=32) as a second policy family.

**Acceptance:** best checkpoint at a sane LIBERO success rate (~50–80% expected; <20% ⇒ debug normalization/camera keys); **spread ≥25 points** between worst and best checkpoint (required for rank correlation to mean anything); failure gallery (20 categorized videos) in `docs/failures.md`; best checkpoint + card on HF.

## T2.2 Ground-truth simulator evaluation

Run the Type 1 harness at scale: each checkpoint × task × **N=50 seeded rollouts**, max 400 steps → `results/sim_success.parquet` with Wilson CIs.

## T2.3 World-model family training

Train all five tiers with identical optimizer/steps-per-example/seed (only the tier config varies). Compute fidelity metrics per tier → `results/wm_fidelity.parquet`.

**Acceptance:** WM-5 gives coherent 30–50-step rollouts at 128px (source of the demo GIF, via `scripts/make_gif.py`); fidelity is monotonic-ish across tiers (if WM-2 beats WM-4, the tier design is broken — fix before proceeding); WM-4/WM-5 + cards on HF.

## T2.4 Success classifier (real scale)

Train on the thousands of labeled simulator videos produced by T2.2. **Require ≥90% held-out accuracy before use**; publish the confusion matrix in `docs/`. Its accuracy bounds every downstream claim — report it in the paper.

## T2.5 Dream rollouts

For every (policy checkpoint × task × WM tier): N=50 dreams, T=200 dreamed steps, scored by the classifier → `results/dream_success.parquet`. Budget check: one triple should evaluate in <10 min.

## T2.6 The calibration study — the headline

1. Per WM tier: **Spearman ρ** between dream and sim success across {P1…P5}, per task and pooled, bootstrap 95% CIs.
2. Task-level Pearson for the best policy.
3. Repeat (1) on held-out tasks.
4. **Trust-region chart:** x = world-model fidelity (16-step LPIPS or divergence step), y = ranking reliability (ρ with CI bars); two curves (in-distribution vs. held-out). Written conclusion states the threshold — whatever the numbers are, report them honestly; a flat curve is still a finding.
5. Robustness (≥2 of): N=20 vs 50 rollouts; classifier threshold ±0.1; T=100 vs 200 horizon.

**Acceptance:** `bash scripts/run_study.sh` regenerates the chart from raw parquets; `LIMITATIONS.md` covers sim-only, small models, classifier bound, single embodiment, LIBERO-specific.

## T2.7 Release (report, Space, HF, repo v1.0)

- **Report** (6–10 pp, arXiv style): cites WorldEval/WPE/Ctrl-World/SIMPLER in the intro; method; tier design; trust-region results; robustness; limitations; reproducibility appendix (commands, GPU-hours, cost).
- **Gradio Space:** dropdowns (task, tier, checkpoint) playing **pre-computed** sim-vs-dream videos side-by-side + interactive trust-region chart. Pre-compute everything; nothing runs live.
- **HF artifacts, all cross-linked:** `dreamgrasp-libero` (already live), `dreamgrasp-vla`, `dreamgrasp-worldmodel`, `dreamgrasp-demo`.
- **Repo v1.0:** README leads with the money GIF + trust-region chart and cites prior work in paragraph 1; CI green; `CITATION.cff`; Apache-2.0; tag `v1.0`.

## T2.8 Definition of done

A stranger can clone the repo, run `bash scripts/reproduce.sh` (with documented GPU access), and regenerate the trust-region chart; all four HF artifacts are live and cross-linked; the report answers "how good must a world model be to trust its evaluations?" with a number and a confidence interval; and the README correctly positions the work relative to WorldEval, WPE, and Ctrl-World.

---

## Repository layout (created in T1.0)

```
dreamgrasp/
├── README.md
├── LICENSE                     # Apache-2.0
├── CITATION.cff
├── LIMITATIONS.md
├── pyproject.toml              # pinned deps; [cuda] extras; ruff/mypy config
├── configs/
│   ├── norm_stats.json
│   ├── policy/smolvla_libero.yaml, smolvla_tiny.yaml
│   └── world_model/tier_{1..5}.yaml, tiny.yaml
├── dreamgrasp/
│   ├── data/        # convert_libero.py, loader.py, stats.py
│   ├── policy/      # train.py, lora.py (optional, cuda-guarded)
│   ├── world_model/ # vae.py, dynamics.py, train.py, fidelity.py
│   ├── eval/        # sim_eval.py, dream_eval.py, success_classifier.py, correlate.py
│   └── utils/       # device.py, video.py, seeding.py, hub.py
├── scripts/         # reproduce.sh, train_policy.sh, train_wm_tier.sh, run_study.sh, make_gif.py
├── tests/           # test_data.py, test_norm.py, test_shapes.py, test_splits.py, test_no_cuda_literals.py
├── notebooks/       # analysis.ipynb
├── docs/            # datasets.md, macos.md, failures.md
├── report/          # report.md → report.pdf, figures/
└── space/           # app.py, precomputed/
```

**Engineering rules:** every experiment = one config + one W&B run; global seeding everywhere; no notebook-only logic; commits tell the phase story.

## GPU budget (Type 2 only — Type 1 costs $0 beyond the Mac)

| Type 2 task | GPU-hours (est.) |
|---|---|
| Parity check + policy fine-tune + sim eval | 40–60 |
| World-model family (5 tiers) | 50–80 |
| Success classifier | 5 |
| Dream rollouts (5 tiers × 5 ckpts × tasks × 50) | 20–30 |
| **Total** | **~120–180 h ≈ $150–400** |

**If over budget, cut in this order:** 3 WM tiers (10%/50%/100%) → 64px latents → 2 task suites → N=30 rollouts. Never cut: the fixed splits, the CIs, or the held-out tasks.