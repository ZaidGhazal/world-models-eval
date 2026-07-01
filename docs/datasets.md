# Dataset notes

## dreamgrasp-libero (processed)

- Source: LIBERO demo HDF5s for `libero_spatial`, `libero_object`, `libero_goal`
  (HF mirror `yifengzhu-hf/LIBERO-datasets`, ~19 GB raw). LIBERO is MIT-licensed — cite
  arXiv:2306.03310.
- Converted with `python -m dreamgrasp.data.convert_libero` (one command) to LeRobotDataset v3:
  **1500 episodes, 200,485 frames, ~600 MB** (AV1-encoded video).
- Published at `world-model-eval/dreamgrasp-libero`.

### Schema

| Field | Spec |
|---|---|
| `observation.images.agentview` | 128×128 RGB video |
| `observation.images.wrist` | 128×128 RGB video |
| `observation.state` | float32 (8,) = eef pos (3) + axis-angle ori (3) + gripper (2) |
| `action` | float32 (7,) delta-EEF + gripper, normalized to [-1,1] per-dim from **train-split** stats |
| `task` | language instruction |

**Resolution deviation from the guide:** the guide's table says 256×256, but LIBERO's raw demos
were collected at 128×128. Upscaling would fabricate pixels and 4× the storage, so we keep native
128×128 (matches the world model's 128px input; SmolVLA upscales internally regardless). Raw
frames are stored bottom-up (`macros_image_convention: opengl`); the converter flips both cameras
upright.

### Splits (`configs/splits.json`, fixed forever)

- Per suite: last 2 tasks in sorted order are **held-out entirely** (6 tasks total, 300 episodes).
- Remaining tasks: 80% train / 10% val / 10% test per task, deterministic per-task RNG.
- Counts: 960 train / 120 val / 120 test / 300 heldout.
- Leakage guard: every episode carries a sha256 content hash of its action sequence;
  `tests/test_splits.py` asserts no val/test/heldout hash appears in train.

### Normalization (`configs/norm_stats.json`)

Per-dim min/max computed over **train episodes only**; `normalize`/`denormalize` round-trip is
unit-tested (`tests/test_norm.py`). The same stats file is shared by the policy (via the dataset)
and the world models, and by sim/dream eval to de-normalize policy actions for the env.

## Loader throughput (M1 Pro, single process — re-benchmark in Type 2)

| Mode | Batch | Throughput |
|---|---|---|
| policy (frame + action chunk 8) | bs 8 | ~425 frames/s |
| world model (contiguous clips) | bs 4 × clip 5 | ~825 frames/s |

Both are far above what tiny-scale MPS training consumes; on the GPU machine use
`num_workers=4` (configs already do).
