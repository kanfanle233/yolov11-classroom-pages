from __future__ import annotations

from pathlib import Path


def resolve_repo_root(anchor: Path) -> Path:
    """Find repository root by looking for known top-level folders."""
    start = anchor.resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate
    raise RuntimeError(f"Cannot resolve repository root from anchor: {start}")


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
