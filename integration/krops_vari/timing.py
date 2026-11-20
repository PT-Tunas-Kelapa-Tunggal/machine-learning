"""Standalone CLI entry point for fertilization_timing inside krops_vari / VariEngine."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import REPO


def resolve_timing_dir() -> Path:
    """Resolve the bundled or repository fertilization_timing directory."""
    if env_dir := os.getenv("KROPS_FERTILIZATION_TIMING_DIR"):
        path = Path(env_dir)
        if (path / "fertilization_timing.py").is_file():
            return path
        raise FileNotFoundError(
            f"KROPS_FERTILIZATION_TIMING_DIR does not contain fertilization_timing.py: {path}"
        )

    # 1. Bundled inside VariEngine distribution: <VariEngine>/fertilization_timing
    bundled = Path(__file__).resolve().parent.parent / "fertilization_timing"
    if (bundled / "fertilization_timing.py").is_file():
        return bundled

    # 2. Development repository location
    repo_timing = REPO / "integration" / "v2" / "fertilization_timing"
    if (repo_timing / "fertilization_timing.py").is_file():
        return repo_timing

    raise FileNotFoundError("fertilization_timing module directory not found.")


def main(argv: list[str] | None = None) -> int:
    """Forward CLI arguments untouched to fertilization_timing.main."""
    timing_dir = resolve_timing_dir()
    timing_str = str(timing_dir)
    if timing_str in sys.path:
        sys.path.remove(timing_str)
    sys.path.insert(0, timing_str)

    import fertilization_timing

    fertilization_timing.main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
