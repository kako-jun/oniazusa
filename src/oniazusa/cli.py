"""CLI entry point for oniazusa."""

import argparse
import sys
from pathlib import Path

from oniazusa.filter import apply_kizuato_style


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform photos into Kizuato-style visual novel backgrounds",
    )
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output path")

    args = parser.parse_args()

    if args.input.is_dir():
        out_dir = args.output or args.input / "oniazusa_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [f for f in args.input.iterdir() if f.suffix.lower() in extensions]
        if not files:
            print(f"No image files found in {args.input}", file=sys.stderr)
            sys.exit(1)
        for f in sorted(files):
            out_path = out_dir / f"{f.stem}_kizuato.png"
            apply_kizuato_style(f, out_path)
            print(f"{f.name} -> {out_path.name}")
    else:
        out_path = args.output or args.input.with_stem(f"{args.input.stem}_kizuato")
        apply_kizuato_style(args.input, out_path)
        print(f"{args.input.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
