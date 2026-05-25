#!/usr/bin/env python3
"""Compare PC-98 dithering strategy candidates A–E on the kizuato reference set."""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root = parent of tools/
REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))

from oniazusa.filter import apply_comparison_three_tone_strategies  # noqa: E402

INPUT_DIR = REPO_ROOT / "input" / "kizuato"
OUTPUT_DIR = REPO_ROOT / "output" / "candidate-compare"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Warning: input directory not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in EXTENSIONS)
    if not files:
        print(f"Warning: no image files found in {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(f"Processing {f.name} ...")
        paths = apply_comparison_three_tone_strategies(f, OUTPUT_DIR)
        for p in paths:
            print(f"  -> {p.name}")


if __name__ == "__main__":
    main()
