from __future__ import annotations

from pathlib import Path


def memory_files(root_dir: Path) -> list[str]:
    return [str(root_dir / "memory" / "AGENTS.md")]


def skill_paths(root_dir: Path) -> list[str]:
    return [str(root_dir / "skills")]
