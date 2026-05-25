# Tint Calibration Notes

## Reference images

- `input/kizuato/bg_ref_01.jpg`
- `input/kizuato/bg_ref_02.png`
- `input/kizuato/bg_ref_03.webp`

## How to run

```
uv run python tools/calibrate_tint.py
```

Optional arguments:

```
uv run python tools/calibrate_tint.py --ref-dir input/kizuato --out-dir tools/calibration_out --k 3
```

## Sampling method

Left-edge (leftmost 10%) and right-edge (rightmost 10%) columns only.
Character stand sprites are excluded by avoiding center regions.

## Palette assignment rule

k-means k=3 on Lab space → sort by L\* descending → assign bright / dark / outline.

## Current preset (green) vs reference ΔE76

(Fill in after running calibrate_tint.py)

## Decision log

- 2026-05-25: initial calibration script added; presets unchanged pending visual review
- Update this section each time presets are changed.
