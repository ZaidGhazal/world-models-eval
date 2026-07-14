# DreamGrasp: How Good Must a World Model Be Before You Can Trust It to Evaluate Robot Policies?

*(DreamGrasp technical report — draft; content complete through T2.6, pending final release
polish in T2.7: money GIF, Space demo cross-links, CI status.)*

## Abstract

World-model-based policy evaluation promises cheap, safe benchmarking of robot policies, but
prior work reports results at a single world-model quality point. We train a controlled family
of five world models spanning a range of measured fidelities and dream-evaluate eight policy
checkpoints against each, calibrated against 12,000 real simulator rollouts. Two findings hold
up under scrutiny.
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

Contributions: (1) two calibration failure modes surfaced by a controlled tier study — dream
evaluation fails silently under distribution shift, and evidence is consistent with the success
classifier partly scoring dream coherence rather than task success; (2) the quality→reliability
calibration curve ("trust region") and a fully reproducible single-GPU harness (data pipeline,
policy training, tiered world models, dream evaluation, analysis), reported together with an
honest robustness analysis showing the curve's per-tier ranking is not stable at this study's
checkpoint count.

## 2. Method

### 2.1 Setup: policies, tasks, ground truth
LIBERO spatial/object/goal; SmolVLA fine-tuned to 8 checkpoints (`step_005000`-`step_040000`,
every 5k steps) spanning `27.00` points of real success-rate spread on `libero_spatial`
(the calibration-scoped suite — see LIMITATIONS.md item 6); ground truth = N=50 seeded
simulator rollouts per (checkpoint, task), 12,000 rollouts total across all three suites and
both train/held-out splits.

### 2.2 World-model family
Frame VAE (128px → 16×16 latents) + causal dynamics transformer over (latent, action) tokens.
Five tiers varying data fraction, context length, capacity, and (tier_5 only) an auxiliary LPIPS
loss on the dynamics stage (Table 1). Identical optimizer, steps-per-example, and seed across
tiers.

**Table 1. World-model tier design.**

| Tier | Data fraction | Context | Dynamics | LPIPS weight |
|---|---:|---:|---|---:|
| tier_1 | 0.10 | 2 | 6L / 256d | 0.0 |
| tier_2 | 0.25 | 2 | 6L / 256d | 0.0 |
| tier_3 | 0.50 | 4 | 12L / 512d | 0.0 |
| tier_4 | 1.00 | 4 | 12L / 512d | 0.0 |
| tier_5 | 1.00 | 8 | 12L / 512d | 0.1 |

### 2.3 Dream evaluation
Policy acts on decoded dreamed frames; proprio from a state head on the dynamics model;
T=200 dreamed steps (design point; T=100 rerun in §3.5); success scored by a frozen-SigLIP
video classifier trained on labeled real simulator rollouts (held-out accuracy `0.9817`,
n=2,400 — see LIMITATIONS.md item 3: this bounds every downstream dream-success claim).

### 2.4 Metrics
Fidelity: PSNR/SSIM/LPIPS at horizons {1,8,16,32}, rollout-divergence step (first dreamed step
whose per-frame MSE exceeds 4x the model's own horizon-1 MSE).
Reliability: Spearman ρ (dream vs. sim ranking of the 8 policy checkpoints, per world-model
tier) with bootstrap 95% CIs; task-level Pearson; in-distribution vs. held-out tasks.

### 2.5 Robustness protocol
Pre-specified per the project's implementation guide: at least two of (a) N=20 vs. 50 dreams per
checkpoint/tier, (b) classifier decision threshold ±0.1 around its 0.5 train-time boundary, (c)
T=100 vs. 200 dreamed steps. We report all three (§3.5). (a) and (b) are computed for free from
the design-point data — (a) by subsampling the real, already-collected `seed<20` dreams rather
than issuing a fresh rollout (dream generation is not bit-reproducible across invocations even
at a fixed seed; LIMITATIONS.md item 8), (b) by re-thresholding the same classifier
probabilities. (c) requires an independent rollout at the shorter horizon and is reported
separately.

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

### 3.3 Fidelity across tiers

**Table 2. Fidelity by tier and horizon (PSNR / SSIM / LPIPS), and mean rollout-divergence
step.**

| Tier | h=1 | h=8 | h=16 | h=32 | Divergence step |
|---|---|---|---|---|---:|
| tier_1 | 26.81 / .943 / .134 | 21.08 / .748 / .250 | 19.47 / .692 / .294 | 17.97 / .661 / .320 | 19.81 |
| tier_2 | 27.04 / .939 / .132 | 21.63 / .752 / .245 | 20.28 / .721 / .272 | 18.65 / .698 / .306 | 20.94 |
| tier_3 | 30.28 / .984 / .046 | 26.42 / .955 / .079 | 24.95 / .938 / .096 | 22.54 / .900 / .121 | 20.38 |
| tier_4 | 31.98 / .990 / .030 | 29.32 / .977 / .040 | 27.49 / .965 / .045 | 25.00 / .946 / .056 | 23.88 |
| tier_5 | 31.49 / .989 / .028 | 26.44 / .964 / .077 | 23.84 / .935 / .121 | 21.19 / .883 / .184 | 12.94 |

Every tier's fidelity degrades monotonically with horizon (PSNR and SSIM fall, LPIPS rises,
h=1 to h=32) — this is the basis for §3.2's coherence argument. Across tiers, fidelity is
*not* monotonic with training scale: tier_4 has the best fidelity of the family on every metric
at every horizon, but tier_5 — the largest-data, longest-context tier — has the worst divergence
step of all five (`12.94`, versus `19.81`-`23.88` for tiers 1-4). This was diagnosed directly
(not assumed): side-by-side ground-truth-vs-dreamed frames show tier_5's rollout staying
faithful through early steps and then visibly disintegrating from around step 16 onward, a
genuine autoregressive quality collapse rather than a measurement artifact. The likely
mechanism, not confirmed, is tier_5's auxiliary LPIPS loss term (Table 1) pulling the shared
dynamics weights toward single-step perceptual sharpness at the cost of multi-step latent
stability, compounded by tier_5's doubled context giving the model more of its own accumulating
error to condition on at each step. The project's only hard acceptance gate on this family
(tier_2 divergence step < tier_4's) passes (`20.94 < 23.88`); the full 5-tier ranking is
reported as data rather than gated on, and tier_5 is retained in every downstream analysis
rather than excluded, since its degraded fidelity is itself part of what §3.4-3.5 calibrate
against.

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

### 3.5 Robustness, in full

**Table 3. Per-tier in-distribution Spearman ρ across every condition tested.**

| Tier | Design point (N=50,T=200) | Threshold 0.4 | Threshold 0.6 | N=20 (subsample) | T=100 (fresh rollout) |
|---|---:|---:|---:|---:|---:|
| tier_1 | 0.881 | 0.857 | 0.786 | 0.857 | 0.024 |
| tier_2 | 0.810 | 0.731 | 0.874 | 0.690 | 0.071 |
| tier_3 | 0.595 | 0.405 | 0.619 | 0.405 | 0.571 |
| tier_4 | 0.548 | 0.756 | 0.577 | **-0.143** | 0.190 |
| tier_5 | 0.524 | 0.452 | 0.429 | 0.357 | 0.238 |

Confidence intervals are wide throughout (design-point CIs already span roughly `0.3` to `1.0`
for the top tiers) and every condition here is a legitimate perturbation of the same real
pipeline — no code, ground truth, or classifier changed between them. Threshold ±0.1 is the
mildest perturbation and roughly preserves tier ordering. N=20 preserves direction for four of
five tiers but flips tier_4's sign. T=100 is decisive: tier_1 and tier_2, the design-point
standouts, collapse to statistical noise, while tier_3 becomes the (still wide-CI) frontrunner.

We checked two specific alternative explanations for the T=100 result before accepting it as a
genuine instability finding. **Truncation:** median real steps-to-success on `libero_spatial`
is `104` (mean `117.3`), so a 100-step horizon cuts off more than half of eventual real
successes — a floor effect from that truncation should depress dream-success scores. It does
not: pooled mean dream-success rose from `0.185` to `0.236` at T=100, and every individual tier
rose with it (§3.2). Ruled out. **Outlier checkpoints:** a leave-one-checkpoint-out analysis on
tier_1 and tier_2 found no single exclusion reconciles the T=200 and T=100 rankings — the best
case (tier_1 excluding `step_025000`) raises T=100's rho from `0.024` to only `0.536`, still far
from T=200's `0.881`. Rank comparison shows a more specific pattern: at T=200, tier_1's
dream-rank matches sim-rank exactly for the four clearly-separated bottom checkpoints and only
reorders within the four checkpoints whose real success rates sit within a `23.2%`-`29.0%` band
(plausibly within Wilson-CI noise at n=50 sim rollouts each); the T=100 collapse goes further,
inverting individual checkpoints outright (§3.2's `step_025000` example) rather than merely
reshuffling an already-ambiguous cluster. Partially, not fully, an outlier-driven effect.

Task-level Pearson at the best checkpoint is the one reliability statistic that holds up across
every condition: `-0.492` (T=200, N=50), `-0.297` (T=200, N=20), `-0.452` (T=100, N=50) — always
negative, always moderate in magnitude.

As presented, this study cannot support a confident ranking of these five world-model tiers by
dream-based reliability: with only 5 tiers and 8 checkpoints, Spearman ρ has too little
statistical power for its point estimate to be trustworthy at any single configuration. Making
the ranking claim testable would require either substantially more policy checkpoints and/or
seeds per checkpoint than this design's fixed budget provides, or a direct audit of the
classifier itself — scoring matched pairs of coherent-but-failed and incoherent-but-successful
dreams to measure the coherence/success conflation from §3.2 directly rather than inferring it.

## 4. Limitations

Sim-only ground truth; small models; classifier accuracy bounds; single embodiment;
LIBERO-specific; calibration scoped to `libero_spatial` — full list in LIMITATIONS.md. Three
items are directly load-bearing for this report's claims, not general caveats: **item 7** (the
held-out silent-failure result, §3.1) — the mechanism behind the paper's primary finding;
**item 8** (dream rollouts are not bit-reproducible given a fixed seed, root-caused to unset
CUDA determinism flags) — the reason §3.5's N=20 check uses subsampling of real dreams rather
than a fresh rerun; and **item 9** (per-tier reliability is not statistically robust at n=8
checkpoints) — the reason §3.4's trust-region curve is reported as illustrative rather than a
settled ranking.

## 5. Reproducibility

Commands, GPU-hours, cost. `bash scripts/run_study.sh` regenerates the design-point (T=200)
trust-region chart and reliability numbers from `results/sim_success.parquet` and
`results/dream_success.parquet`. The T=100 robustness rollout is a separate artifact,
`results/dream_success_t100.parquet`, produced by `scripts/run_t26_horizon100.sh` and analyzed
via `python -m dreamgrasp.eval.correlate --dream results/dream_success_t100.parquet`. The
threshold-sweep and N=20 robustness checks add no new artifacts — `--threshold-sweep 0.4 0.6`
and `--n-dreams-cap 20` recompute them from the design-point parquet already on disk.

## References

- [worldeval2025] WorldEval: World Model as Real-World Robot Policies Evaluator. arXiv:2505.19017
- [wpe2025] World-model-based Policy Evaluation. arXiv:2506.00613
- [ctrlworld2025] Ctrl-World: A Controllable Generative World Model for Robot Manipulation. arXiv:2510.10125
- [simpler2024] Evaluating Real-World Robot Manipulation Policies in Simulation. arXiv:2405.05941
- [robowmbench2026] RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation. arXiv:2604.19092
- [libero2023] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310
- [smolvla2025] SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844
