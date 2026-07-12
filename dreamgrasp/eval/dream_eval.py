"""Dream rollouts: the policy acts on decoded world-model frames.

  python -m dreamgrasp.eval.dream_eval --policy checkpoints/policy/smolvla_tiny/step_000200 \
      --world-model checkpoints/world_model/tiny --wm-tier tiny --n-dreams 2 --horizon 20

Design choices (per guide, documented):
- Proprio inside the dream comes from the dynamics model's state head (not open-loop integration).
- The world model dreams the agentview camera only; the wrist camera is omitted from the policy
  batch — SmolVLA masks missing cameras natively.
- Initial conditions are seeded from real val-split clips (frames + state), so dreams start on-distribution.

Output: results/dream_success.parquet, schema [checkpoint, task, wm_tier, seed, dream_success_prob].
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dreamgrasp.data.loader import ClipDataset, episodes_for_split
from dreamgrasp.eval.sim_eval import load_policy
from dreamgrasp.utils.device import get_device
from dreamgrasp.utils.paths import checkpoint_slug, slugify
from dreamgrasp.utils.seeding import seed_everything
from dreamgrasp.utils.video import save_mp4
from dreamgrasp.world_model.fidelity import load_world_model
from dreamgrasp.world_model.train import resize

REPO_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def dream_rollout(policy, preprocessor, vae, dyn, cfg, seed_clip, task_lang, horizon, device):
    """Policy-in-the-loop dream. Returns (decoded frames (T,H,W,3) uint8, states)."""
    frames = resize(seed_clip["frames"].unsqueeze(0).to(device), cfg.image_size)
    mu, _ = vae.encode(frames.flatten(0, 1))
    latents = mu.view(1, -1, *mu.shape[1:])[:, : cfg.context]
    actions = list(seed_clip["actions"][: cfg.context - 1].to(device).unsqueeze(1))
    state = seed_clip["states"][cfg.context - 1].to(device)
    policy.reset()
    out_frames = []
    for _ in range(horizon):
        frame = vae.decode(latents[:, -1])  # (1,3,H,W) in [0,1]
        out_frames.append(frame[0].permute(1, 2, 0).cpu().numpy())
        batch = {
            "observation.images.agentview": frame.cpu(),
            "observation.state": state.unsqueeze(0).cpu(),
            "task": task_lang,
        }
        action = policy.select_action(preprocessor(batch))  # normalized action space
        actions.append(action.to(device))
        window = latents[:, -cfg.context :]
        acts = torch.cat(actions[-window.shape[1] :], dim=0).unsqueeze(0).float()
        pred, pred_state = dyn.forward(window, acts)
        latents = torch.cat([latents, pred[:, -1:]], dim=1)
        state = pred_state[0, -1]
    return (np.stack(out_frames) * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--world-model", type=Path, required=True)
    parser.add_argument("--wm-tier", required=True)
    parser.add_argument("--n-dreams", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--split", default="val")
    parser.add_argument("--suite", default=None, help="restrict episode sampling to one LIBERO suite")
    parser.add_argument("--episodes", type=int, default=None, help="cap episodes (tiny runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classifier", type=Path, default=None, help="success classifier checkpoint")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "dream_success.parquet")
    parser.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    device = get_device()
    seed_everything(args.seed)
    policy, preprocessor, _ = load_policy(args.policy, device)
    vae, dyn, cfg = load_world_model(args.world_model, device)
    policy_slug = checkpoint_slug(args.policy)
    import wandb

    run = wandb.init(
        project="world-models-eval",
        name=args.run_name or f"dream_eval_{slugify(args.wm_tier)}_{policy_slug}",
        mode=args.wandb,
        config={
            "policy": str(args.policy),
            "world_model": str(args.world_model),
            "wm_tier": args.wm_tier,
            "n_dreams": args.n_dreams,
            "horizon": args.horizon,
            "split": args.split,
            "suite": args.suite,
            "seed": args.seed,
            "classifier": str(args.classifier) if args.classifier else None,
        },
    )

    episodes = episodes_for_split(args.split, suite=args.suite)
    if args.episodes:
        episodes = episodes[: args.episodes]
    ds = ClipDataset(REPO_ROOT / cfg.dataset_root, clip_len=cfg.context + 1, episodes=episodes)
    # task language per clip comes from the underlying dataset item
    scorer = None
    if args.classifier:
        from dreamgrasp.eval.success_classifier import SuccessClassifier

        scorer = SuccessClassifier.load(args.classifier, device)

    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.n_dreams):
        idx = int(rng.integers(len(ds)))
        clip = ds[idx]
        task_lang = ds.ds[ds.starts[idx]]["task"]
        frames = dream_rollout(policy, preprocessor, vae, dyn, cfg, clip, task_lang, args.horizon, device)
        prob = float(scorer.score_frames(frames)) if scorer else float("nan")
        rows.append(
            {
                "checkpoint": str(args.policy),
                "task": task_lang,
                "wm_tier": args.wm_tier,
                "seed": args.seed + k,
                "dream_success_prob": prob,
            }
        )
        if args.save_videos:
            video_name = f"{slugify(args.wm_tier)}__{policy_slug}__seed{args.seed + k}.mp4"
            save_mp4(frames, REPO_ROOT / "results" / "dream_videos" / video_name)
        run.log({"dream/success_prob": prob, "dream/horizon": args.horizon}, step=k)
        print(f"dream {k}: task='{task_lang}' frames={frames.shape} prob={prob}", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        df = pd.concat([pd.read_parquet(args.out), df], ignore_index=True)
    df.to_parquet(args.out, index=False)
    run.summary["mean_dream_success_prob"] = float(pd.DataFrame(rows)["dream_success_prob"].mean())
    run.summary["n_rows"] = len(rows)
    run.finish()
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
