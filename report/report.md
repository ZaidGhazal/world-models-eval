# How Good Must a World Model Be Before You Can Trust It to Evaluate Robot Policies?

*(World Models Eval technical report — skeleton; numbers filled in during Type 2.)*

## Abstract

World-model-based policy evaluation promises cheap, safe benchmarking of robot policies, but
prior work reports results at a single world-model quality point. We train a controlled family
of five world models of increasing fidelity and measure, at each tier, how reliably dreamed
rollouts rank real policy checkpoints — producing a quality→reliability calibration curve and
an open single-GPU harness. **(headline number + CI here)**

## 1. Introduction

Evaluating robot manipulation policies in the real world is slow and expensive; simulators help
but drift from reality. A recent line of work evaluates policies *inside learned world models*:
WorldEval [worldeval2025] runs policies in a latent video model, WPE [wpe2025] studies
world-model policy evaluation head-on, Ctrl-World [ctrlworld2025] builds controllable
manipulation world models, and SIMPLER [simpler2024] shows simulated evaluation can predict
real-world performance. **We do not claim to invent world-model-based evaluation.** What is
missing is a calibration: *how good does the world model have to be before its verdicts can be
trusted?* We answer this with a controlled tier study on LIBERO [libero2023] with SmolVLA
policies [smolvla2025].

Contributions: (1) the quality→reliability calibration curve ("trust region"); (2) a fully
reproducible single-GPU harness (data pipeline, policy training, tiered world models, dream
evaluation, analysis).

## 2. Method

### 2.1 Setup: policies, tasks, ground truth
LIBERO spatial/object/goal; SmolVLA fine-tuned; checkpoints {P1..P5} spanning ≥25 points of
success rate; ground truth = N=50 seeded simulator rollouts per (checkpoint, task).

### 2.2 World-model family
Frame VAE (128px → 16×16 latents) + causal dynamics transformer over (latent, action) tokens.
Five tiers varying data fraction, context length, and capacity (Table 1). Identical optimizer,
steps-per-example, and seed across tiers.

### 2.3 Dream evaluation
Policy acts on decoded dreamed frames; proprio from a state head on the dynamics model;
T=200 dreamed steps; success scored by a frozen-SigLIP video classifier trained on labeled
simulator rollouts (held-out accuracy: **(fill in — bounds all claims)**).

### 2.4 Metrics
Fidelity: PSNR/SSIM/LPIPS at horizons {1,8,16,32}, rollout-divergence step.
Reliability: Spearman ρ (dream vs. sim ranking of {P1..P5}) with bootstrap 95% CIs; task-level
Pearson; in-distribution vs. held-out tasks.

## 3. Results

### 3.1 Fidelity across tiers *(table + monotonicity check)*
### 3.2 The trust region *(headline figure: fidelity vs. ρ, two curves)*
### 3.3 Held-out tasks *(distribution shift)*
### 3.4 Robustness *(N=20 vs 50; classifier threshold ±0.1; T=100 vs 200)*

## 4. Limitations

Sim-only ground truth; small models; classifier accuracy bounds; single embodiment;
LIBERO-specific — see LIMITATIONS.md.

## 5. Reproducibility

Commands, GPU-hours, cost. `bash scripts/run_study.sh` regenerates the trust-region chart from
raw parquets.

## References

- [worldeval2025] WorldEval: World Model as Real-World Robot Policies Evaluator. arXiv:2505.19017
- [wpe2025] World-model-based Policy Evaluation. arXiv:2506.00613
- [ctrlworld2025] Ctrl-World: A Controllable Generative World Model for Robot Manipulation. arXiv:2510.10125
- [simpler2024] Evaluating Real-World Robot Manipulation Policies in Simulation. arXiv:2405.05941
- [libero2023] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310
- [smolvla2025] SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844
