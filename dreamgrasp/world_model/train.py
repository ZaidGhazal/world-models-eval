"""World-model training: stage 1 trains the frame VAE, stage 2 freezes it and trains dynamics.

python -m dreamgrasp.world_model.train --config configs/world_model/tier_4.yaml
python -m dreamgrasp.world_model.train --tiny
"""

import argparse
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dreamgrasp.data.loader import ClipDataset, episodes_for_split, make_loader
from dreamgrasp.utils.device import get_device
from dreamgrasp.utils.seeding import seed_everything
from dreamgrasp.world_model.dynamics import DynamicsTransformer
from dreamgrasp.world_model.vae import FrameVAE

REPO_ROOT = Path(__file__).resolve().parents[2]


def cosine_lr(step: int, total: int, lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * t))


def build_clips(cfg) -> ClipDataset:
    episodes = episodes_for_split(cfg.split)
    if cfg.get("episodes"):
        episodes = episodes[: cfg.episodes]
    n = max(2, int(len(episodes) * cfg.data_fraction))
    return ClipDataset(REPO_ROOT / cfg.dataset_root, clip_len=cfg.context + 1, episodes=episodes[:n])


def resize(frames: torch.Tensor, size: int) -> torch.Tensor:
    """(B,T,C,H,W) -> resized; no-op if already `size`."""
    b, t = frames.shape[:2]
    if frames.shape[-1] == size:
        return frames
    out = torch.nn.functional.interpolate(frames.flatten(0, 1), size=(size, size), mode="bilinear")
    return out.view(b, t, *out.shape[1:])


def train_stage(model, loader, steps, cfg, device, loss_fn, tag, run):
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-5)
    it = iter(loader)
    losses = []
    for step in range(steps):
        for g in optim.param_groups:
            g["lr"] = cosine_lr(step, steps, cfg.lr, cfg.warmup_steps)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["frames"] = resize(batch["frames"], cfg.image_size)
        loss, parts = loss_fn(batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optim.step()
        losses.append(loss.item())
        if step % cfg.log_every == 0 or step == steps - 1:
            print(f"[{tag}] step {step} loss {loss.item():.5f} {parts}", flush=True)
        run.log({f"{tag}/loss": loss.item(), **{f"{tag}/{k}": v for k, v in parts.items()}})
    first = sum(losses[:10]) / min(10, len(losses))
    last = sum(losses[-10:]) / min(10, len(losses))
    print(f"[{tag}] mean(loss[:10])={first:.5f} mean(loss[-10:])={last:.5f}")
    if last >= first:
        raise SystemExit(f"[{tag}] FAILED: loss did not decrease")
    return first, last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/world_model/tier_4.yaml"))
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config_path = REPO_ROOT / "configs/world_model/tiny.yaml" if args.tiny else Path(args.config)
    cfg = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(args.overrides))
    device = get_device()
    seed_everything(cfg.seed)
    print(f"config={config_path.name} device={device}")

    clips = build_clips(cfg)
    print(f"clips: {len(clips)} (clip_len={cfg.context + 1})")
    loader = make_loader(clips, cfg.batch_size, num_workers=cfg.num_workers)

    import wandb

    run = wandb.init(project="dreamgrasp", name=cfg.name, mode=cfg.wandb, config=OmegaConf.to_container(cfg))
    out_dir = REPO_ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Stage 1: VAE on individual frames (flatten clips).
    vae = FrameVAE(latent_dim=cfg.latent_dim).to(device)

    def vae_loss(batch):
        frames = batch["frames"].flatten(0, 1)
        loss, parts = vae(frames)
        parts.pop("recon")
        return loss, parts

    train_stage(vae, loader, cfg.vae_steps, cfg, device, vae_loss, "vae", run)
    torch.save(vae.state_dict(), out_dir / "vae.pt")

    # Stage 2: freeze VAE, train dynamics on latent clips.
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    lpips_model = None
    if cfg.lpips_weight > 0:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    latent_hw = cfg.image_size // 8
    dyn = DynamicsTransformer(
        latent_dim=cfg.latent_dim,
        latent_hw=latent_hw,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        context=cfg.context,
    ).to(device)
    n_params = sum(p.numel() for p in dyn.parameters())
    print(f"dynamics params: {n_params / 1e6:.1f}M")

    def dyn_loss(batch):
        with torch.no_grad():
            b, t = batch["frames"].shape[:2]
            mu, _ = vae.encode(batch["frames"].flatten(0, 1))
            latents = mu.view(b, t, *mu.shape[1:])
        loss, parts = dyn.loss(latents, batch["actions"], batch["states"])
        if lpips_model is not None:
            pred, _ = dyn.forward(latents[:, :-1], batch["actions"][:, :-1])
            decoded = vae.decode(pred[:, -1])
            target = batch["frames"][:, -1]
            lp = lpips_model(decoded * 2 - 1, target * 2 - 1).mean()
            loss = loss + cfg.lpips_weight * lp
            parts["lpips"] = lp.item()
        return loss, parts

    train_stage(dyn, loader, cfg.dyn_steps, cfg, device, dyn_loss, "dyn", run)
    torch.save(dyn.state_dict(), out_dir / "dynamics.pt")
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    run.finish()


if __name__ == "__main__":
    main()
