# oniazusa Spec

## Command

```bash
oniazusa INPUT [OPTIONS]
```

`INPUT` may be a file or a directory.

Supported extensions in directory mode:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

## Output behavior

- single file mode: defaults to `<stem>_kizuato<suffix>`
- directory mode: defaults to `<input>/oniazusa_out/`

## Options

### `--tint`

Preset tint palette.

Current choices:

- `green`
- `yellow`
- `blue`
- `purple`

### `--levels`

Controls the quantization granularity used during ordered dithering.

- lower values increase posterization
- higher values preserve more tonal variation

### `--pre-blur`

Gaussian blur sigma applied to the grayscale working image before ordered dithering.

This exists because raw gradients usually look harsh when sent directly into a visible grid.

### `--glow`

Glow-like blend ratio applied after pre-blur and before dithering.

This helps create smoother large-scale tonal flow before the Bayer pattern becomes visible.

### `--mode`

Processing mode. Controls which rendering pipeline is used.

- `kizuato` (default): general multi-level Bayer dithering mapped to a tinted palette
- `three-tone`: intentional 3-level palette with bright / dark / outline tones

The `--levels` option applies only to `kizuato` mode. In `three-tone` mode it is ignored.

## Processing pipeline

### kizuato (default)

1. bilateral filtering to flatten texture while keeping edges
2. grayscale blur plus Canny edge extraction
3. soft edge darkening overlay
4. slight blur to blend preprocessed image
5. downscale to a low-resolution working canvas
6. convert to grayscale
7. smooth tonal gradients with Gaussian blur and glow-like blending
8. ordered Bayer dithering with highlight-sensitive strength
9. map grayscale to a tinted palette
10. nearest-neighbor upscale to restore visible pixel structure

### three-tone

1. bilateral filtering to flatten texture while keeping edges
2. grayscale blur plus Canny edge extraction
3. soft edge darkening overlay
4. slight blur to blend preprocessed image
5. downscale to a fixed low-resolution working canvas (scale=0.12)
6. convert to grayscale
7. smooth tonal gradients with Gaussian blur and glow-like blending
8. 3-level quantization at thresholds t1=0.45, t2=0.72 (bright / dark / outline)
9. ordered Bayer dithering applied only within ±0.12 transition bands around each threshold
10. map each level to preset BGR colors (bright_bgr / dark_bgr / outline_bgr)
11. nearest-neighbor upscale to restore visible pixel structure

## Failure behavior

- directory mode with no supported files exits with status 1
- unreadable input raises an error
