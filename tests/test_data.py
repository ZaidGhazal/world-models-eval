"""Converted-dataset integrity (skipped until the dataset exists locally)."""

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DS_ROOT = REPO_ROOT / "data" / "lerobot" / "zaid-ghazal" / "dreamgrasp-libero"
SPLITS = REPO_ROOT / "configs" / "splits.json"
STATS = REPO_ROOT / "configs" / "norm_stats.json"

pytestmark = pytest.mark.skipif(
    not (DS_ROOT / "meta" / "info.json").exists(), reason="converted dataset not present"
)


def test_episode_count_matches_plan():
    info = json.loads((DS_ROOT / "meta" / "info.json").read_text())
    plan = json.loads(SPLITS.read_text())
    assert info["total_episodes"] == len(plan)


def test_actions_normalized_and_states_shaped():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=DS_ROOT.name, root=DS_ROOT, episodes=[0])
    item = ds[0]
    assert item["action"].shape == (7,)
    assert item["observation.state"].shape == (8,)
    actions = np.stack([ds[i]["action"].numpy() for i in range(0, len(ds), 10)])
    assert actions.min() >= -1.0 - 1e-5 and actions.max() <= 1.0 + 1e-5


def test_norm_stats_dims():
    stats = json.loads(STATS.read_text())
    assert len(stats["action"]["min"]) == 7
    assert len(stats["observation.state"]["min"]) == 8
