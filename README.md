# World Models Eval

**How good must a learned world model be before you can trust it to evaluate robot policies?**

World Models Eval is a single-GPU, fully reproducible study measuring the relationship between world-model
quality and the reliability of world-model-based policy evaluation, on LIBERO manipulation tasks with
SmolVLA policies.

This project **builds on** prior work in world-model-based policy evaluation — it does not claim to
invent it:

- **WorldEval** — [arXiv:2505.19017](https://arxiv.org/abs/2505.19017)
- **WPE (World-model-based Policy Evaluation)** — [arXiv:2506.00613](https://arxiv.org/abs/2506.00613)
- **Ctrl-World** — [arXiv:2510.10125](https://arxiv.org/abs/2510.10125)
- **SIMPLER** — [arXiv:2405.05941](https://arxiv.org/abs/2405.05941)

Our contributions are (1) the **quality→reliability calibration curve** ("trust region") and
(2) the **open single-GPU harness** that produces it.

## Status

Type 1 (development, Apple Silicon) in progress. See `IMPLEMENTATION_GUIDE.md` for the full spec.

## Setup (macOS / Apple Silicon)

```bash
brew install ffmpeg git-lfs
conda create -n world-models-eval python=3.10 -y && conda activate world-models-eval
pip install -e ".[dev]"
pip install -e ./third_party/LIBERO --config-settings editable_mode=compat --no-deps
python scripts/smoke_test.py
```

macOS-specific workarounds are documented in [docs/macos.md](docs/macos.md).

## License

Apache-2.0. LIBERO (vendored under `third_party/`) is MIT-licensed — cite
[the LIBERO benchmark](https://arxiv.org/abs/2306.03310) if you use the processed dataset.
