import numpy as np

from dreamgrasp.data.stats import compute_stats, denormalize, normalize


def test_round_trip():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1000, 7)) * np.array([5, 1, 0.1, 2, 3, 0.5, 1])).astype(np.float32)
    stats = compute_stats(x)
    np.testing.assert_allclose(denormalize(normalize(x, stats), stats), x, atol=1e-4)


def test_range_and_constant_dim():
    x = np.stack([np.linspace(-3, 9, 100), np.zeros(100)], axis=1).astype(np.float32)
    stats = compute_stats(x)
    n = normalize(x, stats)
    assert n.min() >= -1.0 - 1e-6 and n.max() <= 1.0 + 1e-6
    # constant dim maps to a constant, round-trips exactly
    np.testing.assert_allclose(denormalize(n, stats)[:, 1], 0.0, atol=1e-5)
