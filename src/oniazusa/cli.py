"""CLI entry point for oniazusa."""

import argparse
import sys
from pathlib import Path

from oniazusa.filter import (
    OUTLINE_STRATEGIES,
    PREPROCESS_MODES,
    PRESETS,
    apply_comparison,
    apply_comparison_preprocess,
    apply_kizuato_style,
    apply_three_tone,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform photos into Kizuato-style visual novel backgrounds",
    )
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output path")
    parser.add_argument(
        "-t",
        "--tint",
        default="green",
        choices=list(PRESETS.keys()),
        help="Color tint preset (default: green)",
    )
    parser.add_argument(
        "-l",
        "--levels",
        type=int,
        default=16,
        help="Number of dithering levels (default: 16)",
    )
    parser.add_argument(
        "--pre-blur",
        type=float,
        default=1.4,
        help="Gaussian blur sigma before grid dithering (default: 1.4)",
    )
    parser.add_argument(
        "--glow",
        type=float,
        default=0.18,
        help="Glow-like smoothing blend before grid dithering, 0.0-1.0 (default: 0.18)",
    )
    parser.add_argument(
        "--mode",
        default="kizuato",
        choices=["kizuato", "three-tone"],
        help="Processing mode (default: kizuato)",
    )
    parser.add_argument(
        "--outline-strategy",
        default="edge-overlay",
        choices=OUTLINE_STRATEGIES,
        dest="outline_strategy",
        help="Outline rendering strategy for kizuato mode (default: edge-overlay)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all 4 outline strategies and output individual images plus a collage"
        " (kizuato mode only)",
    )
    parser.add_argument(
        "--preprocess",
        default="none",
        choices=PREPROCESS_MODES,
        help="Preprocessing mode before main pipeline (default: none)",
    )
    parser.add_argument(
        "--compare-preprocess",
        action="store_true",
        dest="compare_preprocess",
        help="Run all 4 preprocess modes and output individual images plus a collage",
    )

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
            if args.compare_preprocess:
                paths = apply_comparison_preprocess(
                    f,
                    out_dir,
                    tint=args.tint,
                    levels=args.levels,
                    pre_blur_sigma=args.pre_blur,
                    glow_strength=args.glow,
                )
                for p in paths:
                    print(f"{f.name} -> {p.name}")
            elif args.mode == "three-tone":
                out_path = out_dir / f"{f.stem}_three_tone.png"
                apply_three_tone(
                    f,
                    out_path,
                    tint=args.tint,
                    pre_blur_sigma=args.pre_blur,
                    glow_strength=args.glow,
                    preprocess=args.preprocess,
                )
                print(f"{f.name} -> {out_path.name}")
            elif args.compare:
                paths = apply_comparison(
                    f,
                    out_dir,
                    tint=args.tint,
                    levels=args.levels,
                    pre_blur_sigma=args.pre_blur,
                    glow_strength=args.glow,
                )
                for p in paths:
                    print(f"{f.name} -> {p.name}")
            else:
                out_path = out_dir / f"{f.stem}_kizuato.png"
                apply_kizuato_style(
                    f,
                    out_path,
                    tint=args.tint,
                    levels=args.levels,
                    pre_blur_sigma=args.pre_blur,
                    glow_strength=args.glow,
                    outline_strategy=args.outline_strategy,
                    preprocess=args.preprocess,
                )
                print(f"{f.name} -> {out_path.name}")
    else:
        if args.compare_preprocess:
            out_dir = args.output or args.input.parent / "oniazusa_out"
            paths = apply_comparison_preprocess(
                args.input,
                out_dir,
                tint=args.tint,
                levels=args.levels,
                pre_blur_sigma=args.pre_blur,
                glow_strength=args.glow,
            )
            for p in paths:
                print(f"{args.input.name} -> {p.name}")
        elif args.mode == "three-tone":
            out_path = args.output or args.input.with_stem(f"{args.input.stem}_three_tone")
            apply_three_tone(
                args.input,
                out_path,
                tint=args.tint,
                pre_blur_sigma=args.pre_blur,
                glow_strength=args.glow,
                preprocess=args.preprocess,
            )
            print(f"{args.input.name} -> {out_path.name}")
        elif args.compare:
            out_dir = args.output or args.input.parent / "oniazusa_out"
            paths = apply_comparison(
                args.input,
                out_dir,
                tint=args.tint,
                levels=args.levels,
                pre_blur_sigma=args.pre_blur,
                glow_strength=args.glow,
            )
            for p in paths:
                print(f"{args.input.name} -> {p.name}")
        else:
            out_path = args.output or args.input.with_stem(f"{args.input.stem}_kizuato")
            apply_kizuato_style(
                args.input,
                out_path,
                tint=args.tint,
                levels=args.levels,
                pre_blur_sigma=args.pre_blur,
                glow_strength=args.glow,
                outline_strategy=args.outline_strategy,
                preprocess=args.preprocess,
            )
            print(f"{args.input.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
