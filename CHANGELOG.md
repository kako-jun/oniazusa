# Changelog

## v0.1.0 — Initial release

- **`oniazusa` CLI**: transform photos into Kizuato-style visual novel backgrounds
- **`--mode kizuato`** (default): ordered Bayer-matrix dithering with tint presets (green / yellow / blue / purple)
- **`--mode three-tone`**: explicit 3-level palette (bright / dark / outline) with transition-band dithering
- **`--outline-strategy`**: four outline treatments to compare side-by-side
  - `edge-overlay` (default): Canny edges darkened 50% on full-resolution image
  - `edge-bias`: edge proximity biases the grayscale field darker at low resolution
  - `dither-density`: edge proximity amplifies dither strength
  - `threshold-shift`: quantisation threshold is shifted near edges
- **`--compare`**: runs all four outline strategies and saves individual images plus a side-by-side collage
- **`tools/calibrate_tint.py`**: sample left/right-edge colors from reference images, cluster with k-means, and compare against current presets via ΔE76
- Batch directory mode supported for all filter modes
