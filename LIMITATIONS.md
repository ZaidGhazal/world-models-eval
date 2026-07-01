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
