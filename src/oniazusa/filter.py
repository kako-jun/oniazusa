"""Kizuato-style image filter."""

from pathlib import Path

import cv2
import numpy as np

# 8x8 Bayer matrix for ordered dithering (screen tone pattern)
BAYER_8X8 = np.array([
    [ 0, 48, 12, 60,  3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [ 8, 56,  4, 52, 11, 59,  7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [ 2, 50, 14, 62,  1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58,  6, 54,  9, 57,  5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
], dtype=np.float32) / 64.0


# Preset tint colors (BGR) - very light, like tinted white paper
PRESETS = {
    "green":  (210, 240, 200),   # barely-green white paper
    "yellow": (200, 235, 245),   # barely-yellow white paper
    "blue":   (240, 215, 195),   # barely-blue white paper
    "purple": (100, 60, 80),     # night purple (the only dark one)
}


def apply_kizuato_style(
    input_path: Path,
    output_path: Path,
    tint: str = "green",
    levels: int = 16,
    scale: float = 0.12,
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
    #    a. Bilateral filter x3: flatten textures, keep edges (cel-shading)
    for _ in range(3):
        img = cv2.bilateralFilter(img, 9, 75, 75)

    #    b. Edge detection for clean manga-style outlines (Canny)
    gray_for_edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_for_edges = cv2.GaussianBlur(gray_for_edges, (5, 5), 1.0)
    edges = cv2.Canny(gray_for_edges, 30, 100)

    #    c. Overlay outlines softly (not full black, semi-transparent)
    edge_mask = (edges > 0).astype(np.float32) * 0.5  # 50% opacity
    img_f = img.astype(np.float32)
    for c in range(3):
        img_f[:, :, c] *= (1.0 - edge_mask)
    img = img_f.astype(np.uint8)

    #    d. Final blur to blend
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    # 2. Downscale to low resolution
    small_w = int(orig_w * scale)
    small_h = int(orig_h * scale)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 3. Ordered dithering with Bayer matrix
    h, w = gray.shape
    bayer = np.tile(BAYER_8X8, (h // 8 + 1, w // 8 + 1))[:h, :w]

    # Dither strength varies: full in darks, fades out in highlights
    # Bright areas become smooth gradient, dark areas show screen tone
    dither_strength = np.clip(1.0 - gray * 1.2, 0, 1)  # 0 at bright, 1 at dark
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
