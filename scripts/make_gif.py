"""Make the README money GIF: side-by-side sim vs dream rollout.

python scripts/make_gif.py --sim results/videos/<x>.mp4 --dream results/dream_videos/<y>.mp4
"""

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--dream", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/figures/money.gif"))
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    sim = np.stack(iio.mimread(args.sim, memtest=False))
    dream = np.stack(iio.mimread(args.dream, memtest=False))
    n = min(len(sim), len(dream))
    h = min(sim.shape[1], dream.shape[1])

    def fit(v):
        import numpy as np

        return np.stack([f[:h, : v.shape[2]] for f in v[:n]])

    frames = np.concatenate([fit(sim), fit(dream)], axis=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(args.out, list(frames), fps=args.fps, loop=0)
    print(f"wrote {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
