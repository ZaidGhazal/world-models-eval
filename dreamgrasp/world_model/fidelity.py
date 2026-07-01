"""World-model fidelity metrics: PSNR/SSIM/LPIPS at horizons {1,8,16,32} + rollout-divergence step.

  python -m dreamgrasp.world_model.fidelity --checkpoint checkpoints/world_model/tiny \
      --split val --n-clips 8 [--horizons 1 8]

Output: one row per (horizon, metric) appended to results/wm_fidelity.parquet.
Divergence step: first dream step where per-frame MSE against ground truth exceeds
`--divergence-thresh` (default 4x the horizon-1 MSE of the model itself).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from dreamgrasp.data.loader import ClipDataset, episodes_for_split
from dreamgrasp.utils.device import get_device
from dreamgrasp.world_model.dynamics import DynamicsTransformer
from dreamgrasp.world_model.vae import FrameVAE

REPO_ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (1, 8, 16, 32)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Global-statistics SSIM on [0,1] images (B,C,H,W) — adequate for relative tier ranking."""
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(dim=(2, 3)), b.mean(dim=(2, 3))
    var_a, var_b = a.var(dim=(2, 3)), b.var(dim=(2, 3))
    cov = ((a - mu_a[..., None, None]) * (b - mu_b[..., None, None])).mean(dim=(2, 3))
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return s.mean().item()


def load_world_model(ckpt_dir: Path, device: str):
    cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    vae = FrameVAE(latent_dim=cfg.latent_dim).to(device).eval()
    vae.load_state_dict(torch.load(ckpt_dir / "vae.pt", map_location=device))
    dyn = DynamicsTransformer(
        latent_dim=cfg.latent_dim,
        latent_hw=cfg.image_size // 8,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        context=cfg.context,
    ).to(device)
    dyn.load_state_dict(torch.load(ckpt_dir / "dynamics.pt", map_location=device))
    dyn.eval()
    return vae, dyn, cfg


@torch.no_grad()
def evaluate(
    ckpt_dir: Path,
    split: str,
    n_clips: int,
    horizons,
    divergence_thresh: float | None,
    n_episodes: int | None = None,
):
    device = get_device()
    vae, dyn, cfg = load_world_model(ckpt_dir, device)
    max_h = max(horizons)
    clip_len = cfg.context + max_h
    episodes = episodes_for_split(split)
    if n_episodes:
        episodes = episodes[:n_episodes]
    ds = ClipDataset(REPO_ROOT / cfg.dataset_root, clip_len=clip_len, episodes=episodes)
    if len(ds) == 0:
        raise SystemExit(f"no clips of length {clip_len} in split {split}")
    idxs = np.linspace(0, len(ds) - 1, min(n_clips, len(ds))).astype(int)

    from dreamgrasp.world_model.train import resize

    per_h: dict[int, dict[str, list[float]]] = {h: {"psnr": [], "ssim": [], "mse": []} for h in horizons}
    divergence_steps: list[int] = []
    step_mses: list[list[float]] = []
    for i in idxs:
        item = ds[int(i)]
        frames = resize(item["frames"].unsqueeze(0).to(device), cfg.image_size)
        actions = item["actions"].unsqueeze(0).to(device)
        mu, _ = vae.encode(frames.flatten(0, 1))
        latents = mu.view(1, clip_len, *mu.shape[1:])
        dreamed, _ = dyn.rollout(latents[:, : cfg.context], actions, max_h)
        decoded = vae.decode(dreamed.flatten(0, 1)).view(1, max_h, 3, cfg.image_size, cfg.image_size)
        truth = frames[:, cfg.context :]
        mses = [torch.mean((decoded[:, h] - truth[:, h]) ** 2).item() for h in range(max_h)]
        step_mses.append(mses)
        for h in horizons:
            per_h[h]["psnr"].append(psnr(decoded[:, h - 1], truth[:, h - 1]))
            per_h[h]["ssim"].append(ssim(decoded[:, h - 1], truth[:, h - 1]))
            per_h[h]["mse"].append(mses[h - 1])

    thresh = divergence_thresh or 4.0 * float(np.mean(per_h[min(horizons)]["mse"]))
    for mses in step_mses:
        over = [t for t, m in enumerate(mses) if m > thresh]
        divergence_steps.append(over[0] + 1 if over else max_h)

    rows = []
    lpips_vals = _lpips(ckpt_dir, per_h, horizons, ds, idxs, vae, dyn, cfg, device)
    for h in horizons:
        rows.append(
            {
                "checkpoint": str(ckpt_dir),
                "split": split,
                "horizon": h,
                "psnr": float(np.mean(per_h[h]["psnr"])),
                "ssim": float(np.mean(per_h[h]["ssim"])),
                "lpips": lpips_vals.get(h, float("nan")),
                "divergence_step": float(np.mean(divergence_steps)),
            }
        )
    return pd.DataFrame(rows)


def _lpips(ckpt_dir, per_h, horizons, ds, idxs, vae, dyn, cfg, device) -> dict[int, float]:
    """LPIPS is optional at tiny scale (heavy VGG download); skip gracefully if unavailable."""
    try:
        import lpips as lpips_lib
    except ImportError:
        print("lpips not installed — recording NaN (fine for Type 1 tiny runs)")
        return {}
    from dreamgrasp.world_model.train import resize

    model = lpips_lib.LPIPS(net="alex").to(device).eval()
    max_h = max(horizons)
    out: dict[int, list[float]] = {h: [] for h in horizons}
    with torch.no_grad():
        for i in idxs:
            item = ds[int(i)]
            frames = resize(item["frames"].unsqueeze(0).to(device), cfg.image_size)
            actions = item["actions"].unsqueeze(0).to(device)
            mu, _ = vae.encode(frames.flatten(0, 1))
            latents = mu.view(1, frames.shape[1], *mu.shape[1:])
            dreamed, _ = dyn.rollout(latents[:, : cfg.context], actions, max_h)
            decoded = vae.decode(dreamed.flatten(0, 1))
            truth = frames[:, cfg.context :].flatten(0, 1)
            d = model(decoded * 2 - 1, truth * 2 - 1).view(-1)
            for h in horizons:
                out[h].append(d[h - 1].item())
    return {h: float(np.mean(v)) for h, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-clips", type=int, default=16)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--divergence-thresh", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="cap val episodes (tiny runs)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "wm_fidelity.parquet")
    args = parser.parse_args()

    df = evaluate(
        args.checkpoint, args.split, args.n_clips, args.horizons, args.divergence_thresh, args.episodes
    )
    print(df.to_string(index=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        df = pd.concat([pd.read_parquet(args.out), df], ignore_index=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
