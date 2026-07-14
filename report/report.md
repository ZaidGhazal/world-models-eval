# DreamGrasp: How Good Must a World Model Be Before You Can Trust It to Evaluate Robot Policies?

*(DreamGrasp technical report — skeleton; numbers filled in during Type 2.)*

## Abstract

World-model-based policy evaluation promises cheap, safe benchmarking of robot policies, but
prior work reports results at a single world-model quality point. We train a controlled family
of five world models of increasing fidelity and dream-evaluate eight policy checkpoints against
each, calibrated against 12,000 real simulator rollouts. Two findings hold up under scrutiny.
First, dream-based evaluation fails *silently* under distribution shift: on held-out tasks every
checkpoint has a 0% real success rate, yet the dream pipeline still produces smoothly varying,
checkpoint-differentiated scores that give no indication anything has gone wrong — there is no
drop in confidence to notice. Second, our evidence is consistent with the success classifier
partly scoring dream *coherence* rather than task *success*: shortening the dream horizon raises
scores at every world-model tier even as it gives the compounding dynamics model less time to
visibly degrade, corroborated by a persistently negative dream-vs-sim task-level correlation and
by individual checkpoints inverting rank across horizon choice. Against this backdrop, per-tier
rank correlation between dream and real success is not stable under robustness checks (N=20 vs.
50 seeds; T=100 vs. 200 steps) at our study's n=8 checkpoints, so we report our tier-reliability
curve as an illustration of the method rather than a settled ranking of these five tiers. We
release the full harness — data pipeline, policy training, the five-tier world-model family,
dream evaluation, and analysis — as a single-GPU-reproducible open pipeline.

## 1. Introduction

Evaluating robot manipulation policies in the real world is slow and expensive; simulators help
but drift from reality. A recent line of work evaluates policies *inside learned world models*:
WorldEval [worldeval2025] runs policies in a latent video model, WPE [wpe2025] studies
world-model policy evaluation head-on, Ctrl-World [ctrlworld2025] builds controllable
manipulation world models, SIMPLER [simpler2024] shows simulated evaluation can predict
real-world performance, and RoboWM-Bench [robowmbench2026] evaluates video world models
through embodiment-grounded execution. **We do not claim to invent world-model-based
evaluation.** What is
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

### 3.1 Held-out tasks: dream evaluation fails silently under distribution shift

On the two held-out `libero_spatial` tasks, every one of the 8 policy checkpoints scored 0/50
real simulator successes (800 held-out rollouts total, zero successes) — a hard generalization
wall, not noise. Spearman rho between dream and sim success is therefore mathematically
undefined for every world-model tier: ground truth has zero variance across checkpoints, so
there is nothing to rank correlation against. Meanwhile the dream pipeline does not go quiet or
flag low confidence here — it produces smoothly varying, checkpoint-dependent success
probabilities on these same episodes (e.g. tier_1 ranges `0.26`-`0.51` across checkpoints) that
look exactly like the graded signal it produces on in-distribution tasks where the correlation
is real. A practitioner watching only dreamed rollouts, with no ground-truth channel, would see
confident-looking, differentiated scores and have no signal that the underlying policy has
completely failed to generalize. This is the calibration study's central caution: dream-based
evaluation is not self-diagnosing under distribution shift — its failure mode here is silent,
not a visible drop in confidence.

### 3.2 Evidence for a coherence/success conflation in the classifier

Shortening the dream horizon from T=200 to T=100 raised mean dream-success scores at every
world-model tier, not just some: pooled mean rose `0.185` -> `0.236`, and every individual tier
rose with it (e.g. tier_1 `0.221` -> `0.308`, tier_5 `0.378` -> `0.452`). This is the opposite of
what a naive truncation account predicts — median real steps-to-success on `libero_spatial` is
`104` (mean `117.3`), so a 100-step horizon cuts off more than half of what would eventually be
real successes, and a floor effect from that truncation should push scores *down*, not up.
Instead, every tier's world-model fidelity independently degrades with horizon without exception
(§3.3): PSNR and SSIM fall and LPIPS rises from horizon 1 to 32 for all five tiers. Taken
together, this evidence is consistent with the classifier partly scoring dream *coherence* — how
visually plausible the rollout looks — rather than purely the depicted task *outcome*: a shorter
dream gives the compounding dynamics model less time to visibly diverge, and a more
coherent-looking failure may score higher than a genuinely successful but visually degraded
rollout.

Two further observations corroborate this reading without isolating the mechanism directly.
Task-level Pearson (dream vs. sim success across the 8 in-distribution tasks, at the best
checkpoint) is consistently negative across every condition tested: `-0.492` (T=200, N=50),
`-0.297` (T=200, N=20 subsample), `-0.452` (T=100, N=50) — dream scores and real task success
are, if anything, inversely related task-by-task, which a pure success signal would not produce.
And the single most dramatic checkpoint-level swing in the study is a rank inversion under
horizon change alone: `step_025000`, tier_1's *best* real checkpoint (`29.0%` sim success), is
tier_1's *worst*-scoring dream checkpoint at T=100 (dream rank 1 of 8), while it sat mid-pack
(rank 6 of 8) at T=200.

We have not directly tested the coherence-conflation mechanism — the observations above are
consistent with it but do not isolate it from other explanations. §3.5 proposes a direct test.

### 3.3 Fidelity across tiers *(table + monotonicity check — to be written next)*

### 3.4 The trust region, illustrated — not a ranking

**This curve is not stable and should not be reported as a settled ranking.** At the
design point (N=50, T=200), tier_1/tier_2 show the highest in-distribution reliability
(rho `0.881`/`0.810`) and reliability appears to *degrade* with tier sophistication
(tier_3/4/5: `0.595`/`0.548`/`0.524`) — not monotonic with fidelity. But neither robustness
axis in the design (§3.5) reproduces this: subsampling to N=20 flips tier_4's sign
(`0.548` -> `-0.143`), and an independent T=100 rollout collapses tier_1/tier_2 to
near-zero (`0.024`/`0.071`) while tier_3 becomes the frontrunner. See LIMITATIONS.md item
9: with only 5 tiers and 8 checkpoints, Spearman rho has too little statistical power for
its point estimate to be trustworthy at any single configuration. Report this section as
"the trust-region method, illustrated" rather than "these five tiers, ranked."

### 3.5 Robustness, in full *(N=20 vs 50; classifier threshold ±0.1; T=100 vs 200 — to be written next)*

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
- [robowmbench2026] RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation. arXiv:2604.19092
- [libero2023] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310
- [smolvla2025] SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844
