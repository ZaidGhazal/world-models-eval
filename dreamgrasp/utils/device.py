"""Device selection and precision policy. The ONLY file allowed to name backends."""

import torch

_CUDA = "cuda"  # single source of truth; tests grep for this literal elsewhere


def get_device() -> str:
    if torch.cuda.is_available():
        return _CUDA
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str | None = None) -> torch.dtype:
    """bf16 on CUDA, fp32 on MPS/CPU (MPS half precision is unreliable)."""
    device = device or get_device()
    return torch.bfloat16 if device == _CUDA else torch.float32


def compile_ok(device: str | None = None) -> bool:
    """torch.compile is unreliable on MPS — only enable on CUDA."""
    return (device or get_device()) == _CUDA
