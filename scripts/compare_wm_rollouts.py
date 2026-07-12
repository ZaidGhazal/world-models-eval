"""Side-by-side ground-truth-vs-dreamed rollout comparison for a world-model checkpoint,
plus per-clip step-MSE curves. Used to diagnose autoregressive divergence (e.g. the tier 5
quality-collapse finding in RUN_LOG 2026-07-10): tier 4 stays coherent through horizon 32,
tier 5 visibly disintegrates from step ~16 despite comparable step-1 fidelity.

python scripts/compare_wm_rollouts.py --checkpoint checkpoints/world_model/tier_5 \
    --n-clips 8 --out /tmp/tier_5_rollout_compare.png
"""

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch

from dreamgrasp.data.loader import ClipDataset, episodes_for_split
from dreamgrasp.utils.device import get_device
from dreamgrasp.world_model.fidelity import load_world_model
from dreamgrasp.world_model.train import resize

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEPS = [1, 4, 8, 16, 24, 32]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-clips", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--out", type=Path, default=Path("/tmp/rollout_compare.png"))
    args = parser.parse_args()

    device = get_device()
    vae, dyn, cfg = load_world_model(args.checkpoint, device)
    clip_len = cfg.context + args.horizon
    ds = ClipDataset(REPO_ROOT / cfg.dataset_root, clip_len=clip_len, episodes=episodes_for_split(args.split))
    idxs = np.linspace(0, len(ds) - 1, min(args.n_clips, len(ds))).astype(int)

    all_mses = []
    for i in idxs:
        item = ds[int(i)]
        frames = resize(item["frames"].unsqueeze(0).to(device), cfg.image_size)
        actions = item["actions"].unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(frames.flatten(0, 1))
            latents = mu.view(1, clip_len, *mu.shape[1:])
            dreamed, _ = dyn.rollout(latents[:, : cfg.context], actions, args.horizon)
            decoded = vae.decode(dreamed.flatten(0, 1))
            decoded = decoded.view(1, args.horizon, 3, cfg.image_size, cfg.image_size)
        truth = frames[:, cfg.context :]
        all_mses.append([torch.mean((decoded[:, h] - truth[:, h]) ** 2).item() for h in range(args.horizon)])
        if i == idxs[0]:
            first_decoded, first_truth = decoded, truth

    arr = np.array(all_mses)
    print(f"{args.checkpoint} (context={cfg.context}) n_clips={len(idxs)}")
    print("mean MSE by step:")
    for step in args.steps:
        print(f"  step {step:2d}: {arr[:, step - 1].mean():.6f}")
    thresh = 4.0 * arr[:, 0].mean()
    divergence = [next((t + 1 for t, m in enumerate(row) if m > thresh), args.horizon) for row in all_mses]
    print(f"threshold (4x step1 mse) = {thresh:.6f}  per-clip divergence steps: {divergence}")

    grid_truth = torch.cat([first_truth[0, s - 1] for s in args.steps], dim=-1)
    grid_dream = torch.cat([first_decoded[0, s - 1].clamp(0, 1) for s in args.steps], dim=-1)
    grid = torch.cat([grid_truth, grid_dream], dim=-2)
    img = (grid.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, img)
    print(f"saved visual comparison -> {args.out} (top row=truth, bottom row=dreamed, steps {args.steps})")


if __name__ == "__main__":
    main()
