"""T1.0 smoke test: LIBERO offscreen render, SmolVLA forward pass, W&B logging.

Run: python scripts/smoke_test.py [--skip-wandb]
"""

import argparse
import os
import sys

# Rendering backend from env only — never hard-code MUJOCO_GL (Linux sets egl; macOS uses default).
print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset — platform default>')}")


def smoke_libero_render() -> None:
    import numpy as np

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from dreamgrasp.utils.video import save_mp4

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    frames = []
    for _ in range(30):
        obs, *_ = env.step([0.0] * 7)
        frames.append(obs["agentview_image"][::-1])  # mujoco renders upside down
    env.close()
    out = save_mp4(np.stack(frames), "results/smoke_libero.mp4")
    assert out.stat().st_size > 0
    print(f"[1/3] LIBERO render OK -> {out} ({task.language})")


def smoke_smolvla_forward() -> None:
    import torch

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from dreamgrasp.utils.device import get_device

    device = get_device()
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device).eval()
    cfg = policy.config
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, "lerobot/smolvla_base", preprocessor_overrides={"device_processor": {"device": device}}
    )
    batch = {"task": "pick up the black bowl"}
    for k, ft in cfg.input_features.items():
        batch[k] = torch.rand(1, *tuple(ft.shape))
    with torch.no_grad():
        action = policy.select_action(preprocessor(batch))
    action = postprocessor(action)
    assert action.shape[-1] == cfg.output_features["action"].shape[0]
    print(f"[2/3] SmolVLA forward OK on {device}, action shape {tuple(action.shape)}")


def smoke_wandb() -> None:
    import wandb

    run = wandb.init(project="dreamgrasp", name="smoke-test", tags=["smoke"])
    run.log({"smoke/dummy_metric": 1.0})
    run.finish()
    print("[3/3] W&B logging OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wandb", action="store_true")
    args = parser.parse_args()

    smoke_libero_render()
    smoke_smolvla_forward()
    if args.skip_wandb:
        print("[3/3] W&B skipped")
    else:
        smoke_wandb()
    print("SMOKE TEST PASSED")
    sys.exit(0)
