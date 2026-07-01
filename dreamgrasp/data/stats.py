"""Per-dim [-1, 1] normalization from training-set stats.

Stats live in configs/norm_stats.json and are shared by the policy and world models.
"""

import json
from pathlib import Path

import numpy as np

NORM_STATS_PATH = Path(__file__).resolve().parents[2] / "configs" / "norm_stats.json"
_EPS = 1e-8


def compute_stats(x: np.ndarray) -> dict:
    """Per-dim min/max over a (N, D) array of training-set values."""
    return {"min": x.min(axis=0).tolist(), "max": x.max(axis=0).tolist()}


def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """Map to [-1, 1] per-dim. Dims with zero range map to 0."""
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    return (2.0 * (x - lo) / np.maximum(hi - lo, _EPS) - 1.0).astype(np.float32)


def denormalize(x: np.ndarray, stats: dict) -> np.ndarray:
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    return ((x + 1.0) / 2.0 * np.maximum(hi - lo, _EPS) + lo).astype(np.float32)


def save_stats(stats: dict, path: str | Path = NORM_STATS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats, indent=2))


def load_stats(path: str | Path = NORM_STATS_PATH) -> dict:
    return json.loads(Path(path).read_text())
