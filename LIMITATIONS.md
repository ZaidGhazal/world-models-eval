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
   LIBERO object/goal suites in general.
   <!-- TODO(T2.2 scope decision): add one sentence characterizing what failed object/goal
   rollouts actually look like, from inspection of saved rollout videos on the GPU host. -->

