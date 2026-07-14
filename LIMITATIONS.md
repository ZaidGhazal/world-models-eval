# Limitations

*(Expanded with numbers during Type 2 — the structure is fixed now.)*

1. **Sim-only ground truth.** "Reliability" here means agreement with a *simulator*, not the real
   world. SIMPLER (arXiv:2405.05941) shows sim can predict real rankings, but our curve calibrates
   dream-vs-sim, not dream-vs-reality.
2. **Small models.** SmolVLA (~450M) policies and 5–80M-parameter dynamics models; conclusions may
   not transfer to frontier-scale world models.
3. **Classifier bound.** Dream success is scored by a learned video classifier; its held-out
   accuracy (reported in docs/) upper-bounds every downstream reliability claim.
4. **Single embodiment.** Franka Panda in LIBERO only.
5. **LIBERO-specific.** Three suites (spatial/object/goal), tabletop manipulation; held-out tasks
   test within-benchmark shift, not cross-domain shift.
6. **Calibration study scoped to `libero_spatial`.** The core T2.4–T2.6 analysis (success
   classifier, dream rollouts, calibration) uses the `libero_spatial` suite, where the T2.2
   ground-truth evaluation (12,000 rollouts: 8 checkpoints x 30 tasks x 50 seeds) showed a real
   worst-to-best checkpoint spread of 27.00 points on train-split tasks — checkpoints are
   meaningfully distinguishable there. `libero_object` (23.5 points) and `libero_goal`
   (16.75 points) did not reach the 25-point spread bar within this run's 40k-step training
   budget, even after removing their hardest tasks, so rank-correlation claims would not be
   supportable on those suites and they are excluded from the gating analysis. Their results
   remain in `results/sim_success.parquet` and are reported in full. This is a property of this
   specific SmolVLA fine-tuning run and its training budget, not a claim about SmolVLA or the
   LIBERO object/goal suites in general. Inspection of sampled failure videos (18 rollouts
   spanning 6 tasks and the 5k/20k/40k checkpoints) shows the dominant modes are grasp
   failure, not scene misunderstanding: early checkpoints sweep the arm over the workspace
   without contacting the task object, while mid/late checkpoints approach the correct region
   but hover above the object, close the gripper on nothing, and then stall or proceed to the
   goal location (basket, stove, drawer) empty-handed; occasionally the correct object is
   engaged but displaced past the target (e.g. a plate dragged off the counter). All sampled
   failures ran the full 400 steps with the scene largely undisturbed — no dropped-object or
   catastrophic cases were observed.
7. **Dream evaluation fails silently under distribution shift.** On the two held-out
   `libero_spatial` tasks, every one of the 8 policy checkpoints scored 0/50 real simulator
   successes (800 rollouts, zero successes — a hard generalization wall, not noise), so
   Spearman rho between dream and sim success is mathematically undefined there for every
   world-model tier: ground truth has zero variance to rank against. The dream pipeline does
   not surface this — it produces smoothly varying, checkpoint-dependent success scores on
   these same held-out episodes (e.g. tier_1 ranges 0.26-0.51 across checkpoints) that look
   exactly like the graded, informative signal it produces in-distribution. A practitioner
   relying only on dreamed rollouts would see confident, differentiated scores with no
   indication that the underlying policy has completely failed to generalize. See
   `report/report.md` section 3.3.
8. **Dream rollouts are not bit-reproducible.** `seed_everything()` does not set CUDA
   determinism flags (`torch.backends.cudnn.deterministic` / `torch.use_deterministic_algorithms`),
   so rerunning `dream_eval.py` with the same seed reproduces the episode/task selection
   exactly but not the final classifier-scored outcome — tiny per-step floating-point
   differences compound over the 200-step autoregressive rollout into effectively
   uncorrelated results (confirmed empirically 2026-07-14; no code changed between the
   original run and the mismatched rerun). Existing dream data remains valid as real,
   independently-generated observations; it just cannot be regenerated bit-for-bit. Any
   N=20/T=100 robustness variant of this study therefore uses subsamples or fresh rollouts
   of the real data rather than attempting to reproduce a specific prior run.
9. **Per-tier reliability estimates are not robust at this checkpoint count.** The
   headline in-distribution result (tier_1/tier_2 clearly the most reliable, degrading
   with tier sophistication: rho `0.881`/`0.810` vs. `0.595`/`0.548`/`0.524` at N=50,
   T=200) does not survive either robustness axis in RUNBOOK's own check. Under N=20
   (subsampling the same real dreams) tier_4's rho flips sign (`0.548` -> `-0.143`). Under
   T=100 (a fresh, independent rollout) tier_1 and tier_2 -- the standout performers at
   T=200 -- collapse to near-zero (`0.024`, `0.071`), while tier_3 becomes the (still
   wide-CI) frontrunner. Both checks used the same real ground truth, the same classifier,
   and the same code path; nothing here points to a bug. The likely explanation is
   structural: with only 5 world-model tiers and 8 policy checkpoints, Spearman rho has
   very little statistical power, and its point estimate is highly sensitive to which
   samples/horizon happen to be used. **This study cannot support a confident claim about
   which world-model tier ranks policies most reliably** — that would require materially
   more policy checkpoints than this design's fixed 5-tier/8-checkpoint budget provides.
   The trust-region chart should be read as illustrative of the method, not as a settled
   ranking of these five tiers.

