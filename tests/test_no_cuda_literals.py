"""Handoff-gate rule: no "cuda" string literals outside dreamgrasp/utils/device.py."""

from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "dreamgrasp"


def test_no_cuda_literals_outside_device_py():
    offenders = []
    for path in PKG.rglob("*.py"):
        if path.name == "device.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if '"cuda"' in line or "'cuda'" in line:
                offenders.append(f"{path.relative_to(PKG.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "cuda literals found:\n" + "\n".join(offenders)
