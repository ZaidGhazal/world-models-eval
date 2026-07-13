"""The correlation pipeline must recover a known correlation from synthetic data (T1.6/T1.7 gate)."""

import pandas as pd

from dreamgrasp.eval.correlate import make_synthetic, normalize_task, ranking_reliability


def test_recovers_high_correlation():
    sim, dream = make_synthetic(1.0, seed=1)
    rho, lo, hi = ranking_reliability(sim, dream)["synthetic"]
    assert rho > 0.9, f"expected near-perfect recovery, got {rho}"


def test_recovers_low_correlation():
    sim, dream = make_synthetic(0.0, seed=2)
    rho, lo, hi = ranking_reliability(sim, dream)["synthetic"]
    assert abs(rho) < 0.6, f"expected weak correlation, got {rho}"


def test_ordering():
    """Higher target mixing weight -> higher recovered rho (monotone sanity)."""
    rhos = []
    for target in (0.0, 0.5, 1.0):
        sim, dream = make_synthetic(target, n_ckpts=8, seed=3)
        rhos.append(ranking_reliability(sim, dream)["synthetic"][0])
    assert rhos[0] < rhos[2], f"rho not increasing with target: {rhos}"


def test_normalize_task_reconciles_sim_and_dream_conventions():
    """sim_eval.py writes LIBERO's underscored slug; dream_eval.py writes the
    LeRobotDataset's natural-language sentence for the same task (2026-07-13 finding:
    this mismatch silently zeroed every checkpoint x task merge)."""
    assert normalize_task("pick_up_the_black_bowl_and_place_it_on_the_plate") == normalize_task(
        "pick up the black bowl and place it on the plate"
    )


def test_ranking_reliability_merges_across_naming_conventions():
    sim = pd.DataFrame(
        [
            {"checkpoint": "A", "task": "pick_up_the_bowl", "seed": 0, "success": True, "steps": 10},
            {"checkpoint": "B", "task": "pick_up_the_bowl", "seed": 0, "success": False, "steps": 400},
        ]
    )
    dream = pd.DataFrame(
        [
            {
                "checkpoint": "A",
                "task": "pick up the bowl",
                "wm_tier": "tier_1",
                "seed": 0,
                "dream_success_prob": 0.9,
            },
            {
                "checkpoint": "B",
                "task": "pick up the bowl",
                "wm_tier": "tier_1",
                "seed": 0,
                "dream_success_prob": 0.1,
            },
        ]
    )
    rho, lo, hi = ranking_reliability(sim, dream)["tier_1"]
    assert rho > 0.99, f"expected the differently-cased/spaced task strings to merge and match, got {rho}"
