"""Dataloaders over the converted LeRobotDataset.

Two batch modes:
  (a) policy training  — single frames with action chunks (delta_timestamps on "action")
  (b) world-model training — contiguous (context+horizon) clips of frames + actions
Split filtering uses configs/splits.json episode indices.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_PATH = REPO_ROOT / "configs" / "splits.json"


def episodes_for_split(split: str, splits_path: Path = SPLITS_PATH) -> list[int]:
    plan = json.loads(Path(splits_path).read_text())
    return [r["episode_index"] for r in plan if r["split"] == split]


def make_policy_dataset(
    root: Path, split: str = "train", chunk_size: int = 8, fps: int = 20, episodes: list[int] | None = None
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = episodes if episodes is not None else episodes_for_split(split)
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(repo_id=root.name, root=root, episodes=episodes, delta_timestamps=delta_timestamps)


class ClipDataset(Dataset):
    """(b) world-model mode: contiguous clips of `clip_len` frames + actions."""

    def __init__(
        self,
        root: Path,
        split: str = "train",
        clip_len: int = 12,
        image_key: str = "observation.images.agentview",
        episodes: list[int] | None = None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        episodes = episodes if episodes is not None else episodes_for_split(split)
        self.ds = LeRobotDataset(repo_id=root.name, root=root, episodes=episodes)
        self.clip_len = clip_len
        self.image_key = image_key
        self.starts: list[int] = []
        # Map episode boundaries -> valid clip start indices (dataset-relative).
        ep_index = self.ds.hf_dataset["episode_index"]
        prev, ep_start = None, 0
        for i, ep in enumerate(list(ep_index) + [None]):
            if ep != prev:
                if prev is not None:
                    self.starts.extend(range(ep_start, max(ep_start, i - clip_len + 1)))
                ep_start = i
                prev = ep

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.starts[idx]
        frames, actions, states = [], [], []
        for t in range(s, s + self.clip_len):
            item = self.ds[t]
            frames.append(item[self.image_key])
            actions.append(item["action"])
            states.append(item["observation.state"])
        return {
            "frames": torch.stack(frames),
            "actions": torch.stack(actions),
            "states": torch.stack(states),
        }


def make_loader(dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True
    )
