"""Kizuato-style image filter."""

from pathlib import Path

import cv2
import numpy as np


def apply_kizuato_style(input_path: Path, output_path: Path) -> None:
    """Transform a photo into a Kizuato-style visual novel background.

    The Kizuato (痕) style is characterized by:
    - Slightly desaturated but not fully grayscale
    - Dark, moody atmosphere with reduced brightness
    - Blue-purple color shift in shadows
    - Soft, slightly blurred look
    - Subtle vignette effect
    """
    img = cv2.imread(str(input_path))
    if img is None:
        msg = f"Could not read image: {input_path}"
        raise ValueError(msg)

    result = img.astype(np.float32)

    # 1. Reduce saturation (keep some color, not fully grayscale)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] *= 0.35  # desaturate to ~35%
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Darken overall
    result *= 0.65

    # 3. Blue-purple tint in shadows
    shadow_mask = (result.mean(axis=2, keepdims=True) < 100).astype(np.float32)
    tint = np.zeros_like(result)
    tint[:, :, 0] = 30.0  # blue channel boost
    tint[:, :, 1] = 5.0   # slight green
    tint[:, :, 2] = 15.0  # slight red (makes purple)
    result += tint * shadow_mask

    # 4. Slight gaussian blur for painterly feel
    result = cv2.GaussianBlur(result, (3, 3), 0.8)

    # 5. Vignette
    h, w = result.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - 0.3 * (distance / max_dist) ** 2
    result *= vignette[:, :, np.newaxis]

    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), result)
