"""Video I/O helpers."""

from pathlib import Path

import imageio.v2 as iio
import numpy as np


def save_mp4(frames: list[np.ndarray] | np.ndarray, path: str | Path, fps: int = 20) -> Path:
    """Save (T, H, W, 3) uint8 frames as MP4 (uses imageio-ffmpeg)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(path, list(np.asarray(frames)), fps=fps, macro_block_size=1)
    return path
