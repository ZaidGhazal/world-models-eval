#!/usr/bin/env bash
# Run T2.2 simulator evaluation over all saved policy checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
task_plan="$(mktemp)"
trap 'rm -f "${task_plan}"' EXIT
python - <<'PY' > "${task_plan}"
import json
import contextlib
import sys
from collections import defaultdict
from pathlib import Path

from libero.libero import benchmark

splits = json.loads(Path("configs/splits.json").read_text())
wanted = {"train", "heldout"}
targets: dict[tuple[str, str], set[str]] = defaultdict(set)
for row in splits:
    if row["split"] in wanted:
        targets[(row["suite"], row["split"])].add(row["task"].removesuffix("_demo"))

for suite_name in ["libero_spatial", "libero_object", "libero_goal"]:
    with contextlib.redirect_stdout(sys.stderr):
        suite = benchmark.get_benchmark_dict()[suite_name]()
    task_ids = {}
    for idx in range(100):
        try:
            task = suite.get_task(idx)
        except Exception:
            break
        task_ids[task.name] = idx
    for split in ["train", "heldout"]:
        names = sorted(targets[(suite_name, split)])
        missing = [name for name in names if name not in task_ids]
        if missing:
            raise SystemExit(f"missing LIBERO task ids for {suite_name}/{split}: {missing}")
        ids = " ".join(str(task_ids[name]) for name in names)
        print(f"{suite_name}\t{split}\t{ids}")
PY

while IFS=$'\t' read -r suite split ids; do
    for ckpt in checkpoints/policy/smolvla_libero/step_*; do
        echo "CHECKPOINT=${ckpt} SUITE=${suite} SPLIT=${split} TASK_IDS=${ids}"
        scripts/sim_eval.sh --checkpoint "${ckpt}" \
            --suite "${suite}" --task-ids ${ids} --split-label "${split}" \
            --n-rollouts 50 --max-steps 400 --video-every 1
    done
done < "${task_plan}"
python -m dreamgrasp.eval.acceptance sim --split train
echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
