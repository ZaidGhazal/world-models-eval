"""The correlation pipeline must recover a known correlation from synthetic data (T1.6/T1.7 gate)."""

from dreamgrasp.eval.correlate import make_synthetic, ranking_reliability


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
