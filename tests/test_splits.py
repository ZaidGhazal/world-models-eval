"""Split integrity: fixed fractions, held-out isolation, no train/heldout leakage by episode hash."""

import json
from pathlib import Path

import pytest

SPLITS_PATH = Path(__file__).resolve().parents[1] / "configs" / "splits.json"

pytestmark = pytest.mark.skipif(not SPLITS_PATH.exists(), reason="splits.json not built yet")


def _plan():
    return json.loads(SPLITS_PATH.read_text())


def test_no_leakage_by_hash():
    plan = _plan()
    train_hashes = {r["hash"] for r in plan if r["split"] == "train"}
    for r in plan:
        if r["split"] in ("val", "test", "heldout"):
            assert r["hash"] not in train_hashes, f"episode {r['episode_index']} leaked into train"


def test_heldout_tasks_fully_excluded():
    plan = _plan()
    heldout_tasks = {r["task"] for r in plan if r["split"] == "heldout"}
    for r in plan:
        if r["task"] in heldout_tasks:
            assert r["split"] == "heldout"


def test_heldout_task_count_per_suite():
    plan = _plan()
    suites = {r["suite"] for r in plan}
    for s in suites:
        heldout = {r["task"] for r in plan if r["suite"] == s and r["split"] == "heldout"}
        assert len(heldout) == 2, f"{s}: expected 2 held-out tasks, got {len(heldout)}"


def test_split_fractions_per_task():
    plan = _plan()
    tasks = {(r["suite"], r["task"]) for r in plan if r["split"] != "heldout"}
    for suite, task in tasks:
        eps = [r for r in plan if r["task"] == task and r["suite"] == suite]
        n_train = sum(r["split"] == "train" for r in eps)
        assert abs(n_train / len(eps) - 0.8) < 0.05


def test_episode_indices_contiguous():
    plan = _plan()
    assert [r["episode_index"] for r in plan] == list(range(len(plan)))
