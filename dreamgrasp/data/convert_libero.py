"""Convert raw LIBERO demo HDF5s to a single LeRobotDataset.

One command: python -m dreamgrasp.data.convert_libero
  --suites libero_spatial libero_object libero_goal
  --repo-id <hf-handle>/world-models-eval

Two passes:
  1. compute per-dim action/state min-max over TRAIN episodes only -> configs/norm_stats.json
  2. write frames (actions normalized to [-1,1]) -> LeRobotDataset at data/lerobot/<repo-id>

Also writes configs/splits.json: per-episode split assignment (train/val/test or heldout task),
with a content hash per episode for the leakage test. Splits are deterministic and fixed forever.

Images are stored at LIBERO's native 128x128 (the guide says 256x256, but the raw demos were
collected at 128x128 — upscaling would fabricate pixels and 4x the storage; documented in
docs/datasets.md).
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from dreamgrasp.data.stats import compute_stats, normalize, save_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_PATH = REPO_ROOT / "configs" / "splits.json"
FPS = 20
N_HELDOUT_TASKS_PER_SUITE = 2
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)  # train / val / test, per task
IMG_SIZE = 128


def task_language(f: h5py.File, path: Path) -> str:
    """Language instruction from the HDF5 problem_info, falling back to the filename."""
    try:
        info = json.loads(f["data"].attrs["problem_info"])
        lang = info.get("language_instruction", "")
        if lang:
            return str(lang).strip('"')
    except (KeyError, json.JSONDecodeError):
        pass
    return path.stem.removesuffix("_demo").replace("_", " ")


def episode_hash(actions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()[:16]


def demo_keys_sorted(f: h5py.File) -> list[str]:
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def build_split_plan(suite_files: dict[str, list[Path]], seed: int = 0) -> list[dict]:
    """Deterministic split assignment. Returns one record per episode, in write order."""
    plan: list[dict] = []
    for suite in sorted(suite_files):
        files = sorted(suite_files[suite])
        # Held-out tasks: last N in sorted order — simple, deterministic, frozen in splits.json.
        heldout = {f.stem for f in files[-N_HELDOUT_TASKS_PER_SUITE:]}
        for path in files:
            with h5py.File(path, "r") as f:
                keys = demo_keys_sorted(f)
                hashes = [episode_hash(f[f"data/{k}/actions"][()]) for k in keys]
            n = len(keys)
            ep_rng = np.random.default_rng(
                int.from_bytes(hashlib.sha256(f"{seed}:{path.stem}".encode()).digest()[:4], "little")
            )
            order = ep_rng.permutation(n)
            n_train, n_val = int(SPLIT_FRACTIONS[0] * n), int(SPLIT_FRACTIONS[1] * n)
            split_of = {}
            for rank, idx in enumerate(order):
                if path.stem in heldout:
                    split_of[idx] = "heldout"
                elif rank < n_train:
                    split_of[idx] = "train"
                elif rank < n_train + n_val:
                    split_of[idx] = "val"
                else:
                    split_of[idx] = "test"
            for i, (k, h) in enumerate(zip(keys, hashes)):
                plan.append(
                    {
                        "suite": suite,
                        "task": path.stem,
                        "file": str(path.relative_to(REPO_ROOT)),
                        "demo_key": k,
                        "split": split_of[i],
                        "hash": h,
                    }
                )
    for ep_idx, rec in enumerate(plan):
        rec["episode_index"] = ep_idx
    return plan


def load_proprio(demo: h5py.Group) -> np.ndarray:
    """Proprio state: ee pos+ori (6) + gripper (2) -> (T, 8) float32."""
    ee = demo["obs/ee_states"][()]
    grip = demo["obs/gripper_states"][()]
    return np.concatenate([ee, grip], axis=1).astype(np.float32)


def compute_train_stats(plan: list[dict]) -> dict:
    actions, states = [], []
    for rec in plan:
        if rec["split"] != "train":
            continue
        with h5py.File(REPO_ROOT / rec["file"], "r") as f:
            demo = f[f"data/{rec['demo_key']}"]
            actions.append(demo["actions"][()].astype(np.float32))
            states.append(load_proprio(demo))
    return {
        "action": compute_stats(np.concatenate(actions)),
        "observation.state": compute_stats(np.concatenate(states)),
    }


def make_features(state_dim: int, action_dim: int) -> dict:
    img = {"dtype": "video", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.agentview": img,
        "observation.images.wrist": img,
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }


def convert(plan: list[dict], stats: dict, repo_id: str, root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first = plan[0]
    with h5py.File(REPO_ROOT / first["file"], "r") as f:
        demo = f[f"data/{first['demo_key']}"]
        state_dim = load_proprio(demo).shape[1]
        action_dim = demo["actions"].shape[1]

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=make_features(state_dim, action_dim),
        root=root,
        robot_type="panda",
    )
    lang_cache: dict[str, str] = {}
    for rec in plan:
        path = REPO_ROOT / rec["file"]
        with h5py.File(path, "r") as f:
            if rec["task"] not in lang_cache:
                lang_cache[rec["task"]] = task_language(f, path)
            lang = lang_cache[rec["task"]]
            demo = f[f"data/{rec['demo_key']}"]
            # macros_image_convention == "opengl": rows stored bottom-up; flip both cameras upright
            agent = demo["obs/agentview_rgb"][()][:, ::-1]
            wrist = demo["obs/eye_in_hand_rgb"][()][:, ::-1]
            state = load_proprio(demo)
            action = normalize(demo["actions"][()].astype(np.float32), stats["action"])
        for t in range(action.shape[0]):
            dataset.add_frame(
                {
                    "observation.images.agentview": agent[t],
                    "observation.images.wrist": wrist[t],
                    "observation.state": state[t],
                    "action": action[t],
                    "task": lang,
                }
            )
        dataset.save_episode()
        if rec["episode_index"] % 50 == 0:
            print(f"episode {rec['episode_index']}/{len(plan)} ({rec['task']})", flush=True)
    dataset.finalize()
    print(f"wrote {len(plan)} episodes to {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "libero_raw")
    parser.add_argument("--repo-id", default="world-models-eval")
    parser.add_argument("--root", type=Path, default=None, help="output dir (default data/lerobot/<repo-id>)")
    parser.add_argument("--max-episodes", type=int, default=None, help="cap for tiny/debug runs")
    args = parser.parse_args()

    suite_files = {s: sorted((args.raw_dir / s).glob("*.hdf5")) for s in args.suites}
    for s, files in suite_files.items():
        if not files:
            raise FileNotFoundError(f"no HDF5s for suite {s} under {args.raw_dir / s}")
        print(f"{s}: {len(files)} task files")

    plan = build_split_plan(suite_files)
    if args.max_episodes:
        plan = plan[: args.max_episodes]
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(plan, indent=1))
    print(f"split plan: {len(plan)} episodes -> {SPLITS_PATH}")

    print("pass 1: computing train-set normalization stats...")
    stats = compute_train_stats(plan)
    save_stats(stats)
    print("pass 2: writing LeRobotDataset...")
    root = args.root or REPO_ROOT / "data" / "lerobot" / args.repo_id
    convert(plan, stats, args.repo_id, root)


if __name__ == "__main__":
    main()
