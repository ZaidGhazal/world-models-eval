# macOS (Apple Silicon) notes & workarounds

Everything in Type 1 runs on an M1 Pro (arm64, MPS, no CUDA). Deviations from the guide's
snippets and every Mac-specific workaround are recorded here.

## Environment

- **arm64-native conda required.** Verified `~/opt/miniconda3` is arm64 (`platform.machine() == "arm64"`).
- **mujoco pinned to 2.3.7, not 3.x.** robosuite 1.4.1 (required by LIBERO; robosuite 1.5 broke
  LIBERO's controller API) calls `mj_fullM` with the mujoco 2.x signature. mujoco 3.10 raises
  `TypeError: mj_fullM(): incompatible function arguments` inside
  `robosuite/controllers/base_controller.py`. mujoco 2.3.7 ships official arm64 wheels; offscreen
  rendering works with the default CGL backend.
- **LIBERO editable install needs legacy mode.** `LIBERO/libero/` has no `__init__.py`, so a PEP 660
  editable install produces an empty import map (`import libero` fails). Install with:
  `pip install -e ./third_party/LIBERO --config-settings editable_mode=compat --no-deps`.
  `--no-deps` is required because LIBERO's `requirements.txt` pins ancient versions
  (torch-era 2022); its `setup.py` has empty `install_requires`, and we supply the runtime deps
  (robosuite, bddl==1.0.1, easydict, gym, matplotlib) from our own pins.
- **bddl must be 1.0.1.** `pip install bddl` today resolves to BDDL 3.x (OmniGibson's), which is a
  different, incompatible package.
- **LIBERO first-import prompt.** `libero.libero.__init__` calls `input()` if `~/.libero/config.yaml`
  doesn't exist. Pre-seed the config (see `scripts/` or README) for non-interactive runs; ours points
  `datasets` at `data/libero_raw/`.

## Rendering

- **Do not set `MUJOCO_GL=egl` on macOS** — that's Linux-only. The default backend (CGL) renders
  offscreen fine. Rendering backend is always read from env, never hard-coded.
- MuJoCo offscreen frames come out vertically flipped; we flip with `[::-1]` at capture time.

## Torch / MPS

- Precision: fp32 on MPS (bf16/fp16 on MPS is unreliable) — enforced by `dreamgrasp/utils/device.py::get_dtype`.
- `torch.compile` disabled unless device is CUDA (`device.py::compile_ok`).
- torch 2.10.0 arm64 wheels; MPS verified with matmul + SmolVLA forward pass.

## API differences vs. the guide's snippets

- lerobot 0.4.4: policy imports live at `lerobot.policies.*` (no `lerobot.common`), and policies
  expect batches run through the pre/post-processor pipeline
  (`lerobot.policies.factory.make_pre_post_processors`) which tokenizes `task` into
  `observation.language.tokens` / attention-mask keys. Raw `{"task": ...}` batches raise `KeyError`.
