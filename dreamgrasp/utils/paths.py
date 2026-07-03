"""Path-safe names for generated artifacts."""

from pathlib import Path


def slugify(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value).strip("_")


def checkpoint_slug(checkpoint: str | Path) -> str:
    path = Path(checkpoint)
    parts = path.parts[-2:] if len(path.parts) >= 2 else path.parts
    return slugify("_".join(parts))
