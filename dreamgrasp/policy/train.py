"""SmolVLA fine-tuning.

  python -m dreamgrasp.policy.train --config configs/policy/smolvla_libero.yaml
  python -m dreamgrasp.policy.train --tiny                      # Mac sanity run
  python -m dreamgrasp.policy.train --tiny --overfit            # overfit-one-batch check

Extra args are OmegaConf dotlist overrides, e.g. `steps=500 batch_size=4`.
"""

import argparse
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dreamgrasp.data.loader import episodes_for_split, make_loader
from dreamgrasp.utils.device import get_device
from dreamgrasp.utils.seeding import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_dataset(cfg):
    import torchvision.transforms.v2 as T
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = episodes_for_split(cfg.split)
    if cfg.episodes:
        episodes = episodes[: cfg.episodes]
    transforms = T.Resize((cfg.image_size, cfg.image_size)) if cfg.image_size else None
    root = REPO_ROOT / cfg.dataset_root
    return LeRobotDataset(
        repo_id=root.name,
        root=root,
        episodes=episodes,
        delta_timestamps={"action": [i / cfg.fps for i in range(cfg.chunk_size)]},
        image_transforms=transforms,
    )


def build_policy(cfg, ds_meta, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained)
    policy_cfg.pretrained_path = cfg.pretrained
    # Adopt OUR dataset's features (7-dof action, 8-dim state, two cameras); the pretrained
    # projections pad to max_state_dim/max_action_dim=32 so weights load without surgery.
    policy_cfg.input_features = {}
    policy_cfg.output_features = {}
    policy_cfg.chunk_size = cfg.chunk_size
    policy_cfg.n_action_steps = cfg.n_action_steps
    policy_cfg.device = device
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        cfg.pretrained,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * t))


def save_checkpoint(policy, preprocessor, postprocessor, out_dir: Path, step: int) -> Path:
    ckpt = out_dir / f"step_{step:06d}"
    policy.save_pretrained(ckpt)
    preprocessor.save_pretrained(ckpt)
    postprocessor.save_pretrained(ckpt)
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/policy/smolvla_libero.yaml"))
    parser.add_argument("--tiny", action="store_true", help="use configs/policy/smolvla_tiny.yaml")
    parser.add_argument("--overfit", action="store_true", help="overfit a single batch")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    config_path = REPO_ROOT / "configs/policy/smolvla_tiny.yaml" if args.tiny else Path(args.config)
    cfg = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(args.overrides))
    device = get_device()
    seed_everything(cfg.seed)
    print(f"config={config_path.name} device={device} steps={cfg.steps} overfit={args.overfit}")

    dataset = build_dataset(cfg)
    print(f"dataset: {dataset.num_episodes} episodes, {len(dataset)} frames")
    loader = make_loader(dataset, cfg.batch_size, num_workers=cfg.num_workers)
    policy, preprocessor, postprocessor = build_policy(cfg, dataset.meta, device)
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-5)

    import wandb

    wandb_cfg = dict(OmegaConf.to_container(cfg))  # type: ignore[arg-type,call-overload]
    run = wandb.init(project="world-models-eval", name=cfg.run_name, mode=cfg.wandb, config=wandb_cfg)

    out_dir = REPO_ROOT / cfg.out_dir
    fixed_batch = None
    it = iter(loader)
    losses: list[float] = []
    t0 = time.time()
    for step in range(cfg.steps):
        for g in optim.param_groups:
            g["lr"] = lr_at(step, cfg)
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            if fixed_batch is not None:
                batch = fixed_batch
            else:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                batch = preprocessor(batch)
                if args.overfit:
                    fixed_batch = batch
            loss, _ = policy.forward(batch)
            (loss / cfg.grad_accum).backward()
            accum_loss += loss.item() / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optim.step()
        losses.append(accum_loss)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(f"step {step} loss {accum_loss:.4f} lr {optim.param_groups[0]['lr']:.2e}", flush=True)
        run.log({"train/loss": accum_loss, "train/lr": optim.param_groups[0]["lr"]}, step=step)
        if (step + 1) % cfg.save_every == 0 or step == cfg.steps - 1:
            ckpt = save_checkpoint(policy, preprocessor, postprocessor, out_dir, step + 1)
            print(f"saved {ckpt}")

    first, last = sum(losses[:10]) / min(10, len(losses)), sum(losses[-10:]) / min(10, len(losses))
    print(f"done in {time.time() - t0:.0f}s: mean(loss[:10])={first:.4f} mean(loss[-10:])={last:.4f}")
    run.summary["loss_first10"], run.summary["loss_last10"] = first, last
    run.finish()
    if args.overfit and last > 0.05:
        raise SystemExit(f"overfit-one-batch FAILED: final loss {last:.4f} > 0.05")
    if last >= first:
        raise SystemExit("FAILED: loss did not decrease")
    print("OK: loss decreased")


if __name__ == "__main__":
    main()
