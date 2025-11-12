#!/usr/bin/env python3
"""CLI wrapper that delegates to the browser-facing implementation."""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parent.parent / "docs" / "compare_interpolation.py"
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
