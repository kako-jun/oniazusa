"""Kizuato-style image filter."""

from pathlib import Path

import cv2
import numpy as np

OUTLINE_STRATEGIES = ["edge-overlay", "edge-bias", "dither-density", "threshold-shift"]
PREPROCESS_MODES = ["none", "denoise", "flatten", "illustration"]

# 8x8 Bayer matrix for ordered dithering (screen tone pattern)
BAYER_8X8 = (
    np.array(
        [
            [0, 48, 12, 60, 3, 51, 15, 63],
            [32, 16, 44, 28, 35, 19, 47, 31],
            [8, 56, 4, 52, 11, 59, 7, 55],
            [40, 24, 36, 20, 43, 27, 39, 23],
            [2, 50, 14, 62, 1, 49, 13, 61],
            [34, 18, 46, 30, 33, 17, 45, 29],
            [10, 58, 6, 54, 9, 57, 5, 53],
            [42, 26, 38, 22, 41, 25, 37, 21],
        ],
        dtype=np.float32,
    )
    / 64.0
)


# Preset tint colors (BGR) - very light, like tinted white paper
PRESETS = {
    # TODO: update after running calibrate_tint.py
    "green": (210, 240, 200),  # barely-green white paper
    "yellow": (200, 235, 245),  # barely-yellow white paper
    "blue": (240, 215, 195),  # barely-blue white paper
    "purple": (100, 60, 80),  # night purple (the only dark one)
}


def _preprocess(img: np.ndarray, mode: str) -> np.ndarray:
    """Apply a preprocessing stage before the main pipeline.

    Args:
        img: BGR uint8 image.
        mode: One of "none", "denoise", "flatten", "illustration".

    Returns:
        Preprocessed BGR uint8 image.
    """
    if mode == "denoise":
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        for _ in range(3):
            img = cv2.bilateralFilter(img, 9, 75, 75)
    elif mode == "flatten":
        for _ in range(3):
            img = cv2.bilateralFilter(img, 9, 150, 150)
    elif mode == "illustration":
        for _ in range(3):
            img = cv2.bilateralFilter(img, 9, 75, 75)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # none: default bilateral x3
        for _ in range(3):
            img = cv2.bilateralFilter(img, 9, 75, 75)
    return img


def _detect_edge_map(img: np.ndarray) -> np.ndarray:
    """Detect edges and return a float32 mask in [0.0, 1.0].

    Args:
        img: BGR uint8 image.

    Returns:
        float32 array of the same H×W, 0.0=non-edge, 1.0=edge.
    """
    gray_for_edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_for_edges = cv2.GaussianBlur(gray_for_edges, (5, 5), 1.0)
    edges = cv2.Canny(gray_for_edges, 30, 100)
    return (edges > 0).astype(np.float32)


def _smooth_tonal_gradient(
    gray: np.ndarray,
    pre_blur_sigma: float,
    glow_strength: float,
) -> np.ndarray:
    """Create a smoother tonal field before ordered dithering.

    The intended look is not raw-photo posterization. We first make the light and shadow flow
    gentler, then place the visible grid-based dither on top of that smoother base.
    """
    smoothed = gray

    if pre_blur_sigma > 0:
        smoothed = cv2.GaussianBlur(smoothed, (0, 0), pre_blur_sigma)

    if glow_strength > 0:
        glow_sigma = max(pre_blur_sigma * 2.5, 1.0)
        glow = cv2.GaussianBlur(smoothed, (0, 0), glow_sigma)
        smoothed = smoothed * (1.0 - glow_strength) + glow * glow_strength

    return np.clip(smoothed, 0.0, 1.0)


def apply_kizuato_style(
    input_path: Path,
    output_path: Path,
    tint: str = "green",
    levels: int = 16,
    scale: float = 0.12,
    pre_blur_sigma: float = 1.4,
    glow_strength: float = 0.18,
    outline_strategy: str = "edge-overlay",
    preprocess: str = "none",
) -> None:
    """Transform a photo into a Kizuato-style visual novel background.

    Process (matching the original 90s technique):
    1. Downscale to low resolution (so pixels/dots are visible)
    2. Convert to grayscale
    3. Ordered dithering (Bayer matrix) to N levels - creates screen tone patterns
    4. Map dithered grayscale to a single-color palette (like colored cellophane)
    5. Upscale back with nearest-neighbor (preserving pixel art look)
    """
    img = cv2.imread(str(input_path))
    if img is None:
        msg = f"Could not read image: {input_path}"
        raise ValueError(msg)

    orig_h, orig_w = img.shape[:2]

    # 1. Pre-process: photo to manga/illustration-like
    img = _preprocess(img, preprocess)

    # Detect edges (full-res) for all strategies
    edge_map = _detect_edge_map(img)

    if outline_strategy == "edge-overlay":
        #    b/c. Edge overlay (existing behavior): darken edges 50% on full-res
        edge_mask = edge_map * 0.5  # 50% opacity
        img_f = img.astype(np.float32)
        for c in range(3):
            img_f[:, :, c] *= 1.0 - edge_mask
        img = img_f.astype(np.uint8)

    #    d. Final blur to blend
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    # 2. Downscale to low resolution
    small_w = int(orig_w * scale)
    small_h = int(orig_h * scale)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Downscale edge map for strategies that operate at low-res.
    # Always computed to avoid possibly-unbound reference; zero-cost for edge-overlay.
    edge_map_small = cv2.resize(edge_map, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )

    # 3. Convert to grayscale, then shape smoother gradients before visible grid dithering.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = _smooth_tonal_gradient(gray, pre_blur_sigma, glow_strength)

    if outline_strategy == "edge-bias":
        gray = np.clip(gray - edge_map_small * 0.2, 0, 1)

    if outline_strategy == "threshold-shift":
        gray = np.clip(gray - edge_map_small * 0.15, 0, 1)

    # 4. Ordered dithering with Bayer matrix
    h, w = gray.shape
    bayer = np.tile(BAYER_8X8, (h // 8 + 1, w // 8 + 1))[:h, :w]

    # Dither strength varies: full in darks, fades out in highlights
    # Bright areas become smooth gradient, dark areas show screen tone
    dither_strength = np.clip(1.0 - gray * 1.15, 0, 1)  # 0 at bright, 1 at dark

    if outline_strategy == "dither-density":
        dither_strength = np.clip(dither_strength + edge_map_small * 0.6, 0, 1)

    dithered = gray + (bayer - 0.5) / levels * dither_strength
    dithered = np.clip(dithered, 0, 1)
    dithered = np.floor(dithered * levels) / levels
    # Blend: highlights use smooth gray, darks use dithered
    dithered = gray * (1.0 - dither_strength) + dithered * dither_strength

    # 4. Map to tint color (colored cellophane effect)
    tint_bgr = PRESETS.get(tint, PRESETS["green"])

    # Black (0,0,0) to tint color at full brightness
    result = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        result[:, :, c] = dithered * tint_bgr[c]

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 5. Upscale back with nearest-neighbor (pixelated look)
    result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(output_path), result)


def apply_comparison(
    input_path: Path,
    output_dir: Path,
    tint: str = "green",
    levels: int = 16,
    pre_blur_sigma: float = 1.4,
    glow_strength: float = 0.18,
) -> list[Path]:
    """Run all 4 outline strategies and save individual images plus a collage.

    Args:
        input_path: Source image path.
        output_dir: Directory to write outputs.
        tint: Color tint preset.
        levels: Dithering quantization levels.
        pre_blur_sigma: Gaussian blur sigma before dithering.
        glow_strength: Glow blend ratio before dithering.

    Returns:
        List of output paths: 4 individual images + 1 collage (5 elements).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    individual_paths: list[Path] = []
    images: list[np.ndarray] = []

    for strategy in OUTLINE_STRATEGIES:
        out_path = output_dir / f"{stem}_compare_{strategy}.png"
        apply_kizuato_style(
            input_path,
            out_path,
            tint=tint,
            levels=levels,
            pre_blur_sigma=pre_blur_sigma,
            glow_strength=glow_strength,
            outline_strategy=strategy,
        )
        individual_paths.append(out_path)
        img = cv2.imread(str(out_path))
        images.append(img)

    # Build collage: align all images to the same height then hstack
    target_h = images[0].shape[0]
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            row_scale = target_h / h
            img = cv2.resize(img, (int(w * row_scale), target_h), interpolation=cv2.INTER_AREA)
        resized.append(img)

    collage = np.hstack(resized)

    # Limit collage width to 4000px
    max_w = 4000
    col_h, col_w = collage.shape[:2]
    if col_w > max_w:
        collage_scale = max_w / col_w
        collage = cv2.resize(
            collage, (max_w, int(col_h * collage_scale)), interpolation=cv2.INTER_AREA
        )

    collage_path = output_dir / f"{stem}_compare.png"
    cv2.imwrite(str(collage_path), collage)

    return [*individual_paths, collage_path]


def apply_comparison_preprocess(
    input_path: Path,
    output_dir: Path,
    tint: str = "green",
    levels: int = 16,
    pre_blur_sigma: float = 1.4,
    glow_strength: float = 0.18,
) -> list[Path]:
    """Run all 4 preprocess modes and save individual images plus a collage.

    Args:
        input_path: Source image path.
        output_dir: Directory to write outputs.
        tint: Color tint preset.
        levels: Dithering quantization levels.
        pre_blur_sigma: Gaussian blur sigma before dithering.
        glow_strength: Glow blend ratio before dithering.

    Returns:
        List of output paths: 4 individual images + 1 collage (5 elements).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    individual_paths: list[Path] = []
    images: list[np.ndarray] = []

    for mode in PREPROCESS_MODES:
        out_path = output_dir / f"{stem}_preprocess_{mode}.png"
        apply_kizuato_style(
            input_path,
            out_path,
            tint=tint,
            levels=levels,
            pre_blur_sigma=pre_blur_sigma,
            glow_strength=glow_strength,
            preprocess=mode,
        )
        individual_paths.append(out_path)
        img = cv2.imread(str(out_path))
        images.append(img)

    # Build collage: align all images to the same height then hstack
    target_h = images[0].shape[0]
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            row_scale = target_h / h
            img = cv2.resize(img, (int(w * row_scale), target_h), interpolation=cv2.INTER_AREA)
        resized.append(img)

    collage = np.hstack(resized)

    # Limit collage width to 4000px
    max_w = 4000
    col_h, col_w = collage.shape[:2]
    if col_w > max_w:
        collage_scale = max_w / col_w
        collage = cv2.resize(
            collage, (max_w, int(col_h * collage_scale)), interpolation=cv2.INTER_AREA
        )

    collage_path = output_dir / f"{stem}_preprocess_compare.png"
    cv2.imwrite(str(collage_path), collage)

    return [*individual_paths, collage_path]


# Three-tone presets: (bright_bgr, dark_bgr, outline_bgr)
THREE_TONE_PRESETS: dict[
    str,
    tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
] = {
    #               bright_bgr        dark_bgr          outline_bgr
    # TODO: update after running calibrate_tint.py
    "green": ((210, 240, 200), (130, 180, 120), (40, 70, 30)),
    "yellow": ((200, 235, 245), (130, 170, 160), (50, 80, 40)),
    "blue": ((240, 215, 195), (140, 160, 200), (40, 60, 120)),
    "purple": ((160, 120, 150), (80, 50, 90), (20, 10, 40)),
}


def apply_three_tone(
    input_path: Path,
    output_path: Path,
    tint: str = "green",
    pre_blur_sigma: float = 1.4,
    glow_strength: float = 0.18,
    preprocess: str = "none",
) -> None:
    """Transform a photo into a 3-tone visual novel background.

    Unlike apply_kizuato_style (general multi-level posterization), this function
    uses an intentional 3-level palette: bright / dark / outline.

    Process:
    1. Bilateral filter x3, Gaussian blur pre-processing (same as kizuato)
    2. Downscale (scale=0.12 fixed)
    3. Grayscale conversion + smooth tonal gradient
    4. 3-level quantization with thresholds t1=0.45, t2=0.72
    5. Ordered dithering only in the transition bands (threshold ±0.12)
    6. Map each level to the preset BGR colors
    7. Nearest-neighbor upscale to restore pixel structure
    """
    img = cv2.imread(str(input_path))
    if img is None:
        msg = f"Could not read image: {input_path}"
        raise ValueError(msg)

    orig_h, orig_w = img.shape[:2]

    # 1. Pre-process: bilateral filter x3, edge overlay, final blur
    img = _preprocess(img, preprocess)

    edge_map = _detect_edge_map(img)
    edge_mask = edge_map * 0.5
    img_f = img.astype(np.float32)
    for c in range(3):
        img_f[:, :, c] *= 1.0 - edge_mask
    img = img_f.astype(np.uint8)

    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    # 2. Downscale
    scale = 0.12
    small_w = int(orig_w * scale)
    small_h = int(orig_h * scale)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # 3. Grayscale + smooth tonal gradient
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = _smooth_tonal_gradient(gray, pre_blur_sigma, glow_strength)

    # 4 & 5. 3-level quantization with ordered dithering in transition bands
    t1, t2 = 0.45, 0.72
    band = 0.12

    h, w = gray.shape
    bayer = np.tile(BAYER_8X8, (h // 8 + 1, w // 8 + 1))[:h, :w]

    # Build a blended gray that uses dithering only in the ±band zones
    # Outside the bands, value is deterministic (no dither noise)
    dithered = gray + (bayer - 0.5) * band

    # Blend factor: 1.0 inside transition band, 0.0 far from threshold
    def _band_weight(g: np.ndarray, threshold: float) -> np.ndarray:
        dist = np.abs(g - threshold)
        return np.clip(1.0 - dist / band, 0.0, 1.0)

    w1 = _band_weight(gray, t1)
    w2 = _band_weight(gray, t2)
    blend = np.clip(w1 + w2, 0.0, 1.0)
    effective = gray * (1.0 - blend) + dithered * blend

    # 3-level assignment
    bright_bgr, dark_bgr, outline_bgr = THREE_TONE_PRESETS.get(tint, THREE_TONE_PRESETS["green"])

    result = np.zeros((h, w, 3), dtype=np.uint8)
    bright_mask = effective >= t2
    dark_mask = (effective >= t1) & ~bright_mask
    outline_mask = ~bright_mask & ~dark_mask

    for c in range(3):
        result[:, :, c] = (
            bright_mask * bright_bgr[c] + dark_mask * dark_bgr[c] + outline_mask * outline_bgr[c]
        ).astype(np.uint8)

    # 7. Nearest-neighbor upscale
    result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(output_path), result)
