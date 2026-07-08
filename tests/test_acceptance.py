"""Unit check for the suite-scoped T2.2 sim acceptance gate."""

import pandas as pd
import pytest

from dreamgrasp.eval.acceptance import check_sim


def make_parquet(tmp_path):
    rows = []
    # spatial has 0.30 spread (passes); goal has 0.10 spread (fails); mixed full scope fails.
    for suite, worst, best in [("libero_spatial", 0.0, 0.30), ("libero_goal", 0.05, 0.15)]:
        for ckpt, rate in [("step_005000", worst), ("step_040000", best)]:
            for seed in range(100):
                rows.append(
                    {
                        "checkpoint": ckpt,
                        "suite": suite,
                        "split": "train",
                        "task": f"{suite}_task",
                        "seed": seed,
                        "success": seed < rate * 100,
                        "steps": 400,
                    }
                )
    path = tmp_path / "sim_success.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_suite_scoped_gate_passes(tmp_path, capsys):
    path = make_parquet(tmp_path)
    check_sim(path, min_best=0.20, min_spread=0.25, split="train", suite="libero_spatial")
    out = capsys.readouterr().out
    assert "all suites (reported, non-gating)" in out
    assert "suite=libero_spatial" in out


def test_full_scope_gate_still_fails(tmp_path):
    path = make_parquet(tmp_path)
    with pytest.raises(SystemExit, match="spread"):
        check_sim(path, min_best=0.20, min_spread=0.25, split="train", suite=None)


def test_unknown_suite_fails(tmp_path):
    path = make_parquet(tmp_path)
    with pytest.raises(SystemExit, match="suite"):
        check_sim(path, min_best=0.20, min_spread=0.25, split="train", suite="libero_object_typo")
