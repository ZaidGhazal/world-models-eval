"""Ground-truth simulator evaluation: seeded LIBERO rollouts of a policy checkpoint.

  python -m dreamgrasp.eval.sim_eval --checkpoint checkpoints/policy/smolvla_tiny/step_000200 \
      --suite libero_goal --task-ids 0 1 --n-rollouts 3 --max-steps 100

Output: results/sim_success.parquet with schema [checkpoint, task, seed, success, steps]
(appends across invocations), plus an MP4 for every `--video-every`-th rollout.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dreamgrasp.data.stats import denormalize, load_stats
from dreamgrasp.eval.metrics import wilson_ci
from dreamgrasp.utils.device import get_device
from dreamgrasp.utils.paths import checkpoint_slug, slugify
from dreamgrasp.utils.seeding import seed_everything
from dreamgrasp.utils.video import save_mp4

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG = 128  # match training-data resolution


def load_policy(checkpoint: Path, device: str):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, str(checkpoint), preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, preprocessor, postprocessor


def obs_to_batch(obs: dict, task_lang: str) -> dict:
    import robosuite.utils.transform_utils as T

    def img(key):
        # env renders bottom-up (opengl convention) -> flip upright, HWC uint8 -> CHW float [0,1]
        arr = np.ascontiguousarray(obs[key][::-1])
        return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    state = np.hstack(
        [obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)
    return {
        "observation.images.agentview": img("agentview_image"),
        "observation.images.wrist": img("robot0_eye_in_hand_image"),
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "task": task_lang,
    }


def rollout(env, policy, preprocessor, postprocessor, task_lang, init_state, max_steps, action_stats):
    env.reset()
    env.set_init_state(init_state)
    policy.reset()
    obs = env.env._get_observations()
    frames, success, steps = [], False, max_steps
    for t in range(max_steps):
        batch = preprocessor(obs_to_batch(obs, task_lang))
        with torch.no_grad():
            action = policy.select_action(batch)
        action = postprocessor(action)
        raw = denormalize(action.cpu().numpy()[0], action_stats)
        obs, _, done, _ = env.step(raw.tolist())
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1]))
        if env.check_success():
            success, steps = True, t + 1
            break
    return success, steps, frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--n-rollouts", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "sim_success.parquet")
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    device = get_device()
    seed_everything(args.seed)
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, device)
    action_stats = load_stats()["action"]
    suite = benchmark.get_benchmark_dict()[args.suite]()
    ckpt_slug = checkpoint_slug(args.checkpoint)

    rows = []
    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # LIBERO init-state files are numpy pickles; torch>=2.6 defaults weights_only=True and
        # numpy 2 renamed np.core -> np._core, so the allowlist can't match. These are local
        # benchmark files vendored in third_party — load them directly as trusted.
        init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        init_states = torch.load(init_path, weights_only=False)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=IMG, camera_widths=IMG)
        env.seed(args.seed)
        n_success = 0
        for k in range(args.n_rollouts):
            seed = args.seed + k
            success, steps, frames = rollout(
                env,
                policy,
                preprocessor,
                postprocessor,
                task.language,
                init_states[seed % len(init_states)],
                args.max_steps,
                action_stats,
            )
            n_success += success
            rows.append(
                {
                    "checkpoint": str(args.checkpoint),
                    "task": task.name,
                    "seed": seed,
                    "success": bool(success),
                    "steps": int(steps),
                }
            )
            if k % args.video_every == 0:
                video_name = f"{ckpt_slug}__{slugify(task.name)}__seed{seed}.mp4"
                save_mp4(
                    np.stack(frames),
                    REPO_ROOT / "results" / "videos" / video_name,
                )
            print(f"{task.name} seed={seed} success={success} steps={steps}", flush=True)
        env.close()
        lo, hi = wilson_ci(n_success, args.n_rollouts)
        print(f"{task.name}: {n_success}/{args.n_rollouts} wilson95=[{lo:.3f},{hi:.3f}]")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        df = pd.concat([pd.read_parquet(args.out), df], ignore_index=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
