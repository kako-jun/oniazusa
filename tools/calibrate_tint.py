"""Calibrate tint palettes against reference scenes.

Samples left-edge (leftmost 10%) and right-edge (rightmost 10%) columns of each
reference image to avoid character stand sprites in the center.

Usage:
    uv run python tools/calibrate_tint.py [--ref-dir DIR] [--out-dir DIR] [--k K]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Repository root = parent of tools/
REPO_ROOT = Path(__file__).resolve().parent.parent

# Import filter module from src/
sys.path.insert(0, str(REPO_ROOT / "src"))
from oniazusa.filter import THREE_TONE_PRESETS, apply_kizuato_style  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_images(ref_dir: Path) -> list[tuple[Path, np.ndarray]]:
    """Load all supported images from ref_dir."""
    results: list[tuple[Path, np.ndarray]] = []
    for p in sorted(ref_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            img = cv2.imread(str(p))
            if img is not None:
                results.append((p, img))
    return results


def _sample_edge_pixels(img: np.ndarray, edge_fraction: float = 0.10) -> np.ndarray:
    """Return pixels from the leftmost and rightmost edge_fraction of columns.

    Args:
        img: BGR uint8 image (H, W, 3).
        edge_fraction: Fraction of width to sample from each side (default 0.10 = 10%).

    Returns:
        2D array of shape (N, 3) in BGR uint8.
    """
    h, w = img.shape[:2]
    edge_cols = max(1, int(w * edge_fraction))

    left = img[:, :edge_cols, :].reshape(-1, 3)
    right = img[:, w - edge_cols :, :].reshape(-1, 3)
    return np.vstack([left, right])


def _bgr_to_lab(pixels_bgr: np.ndarray) -> np.ndarray:
    """Convert (N, 3) BGR uint8 pixels to (N, 3) Lab float32."""
    # cv2.cvtColor expects (H, W, 3); treat N pixels as a 1×N image
    img_1xn = pixels_bgr.reshape(1, -1, 3).astype(np.uint8)
    lab_1xn = cv2.cvtColor(img_1xn, cv2.COLOR_BGR2Lab)
    return lab_1xn.reshape(-1, 3).astype(np.float32)


def _kmeans_lab(pixels_bgr: np.ndarray, k: int) -> list[dict]:
    """Run k-means on Lab pixels and return cluster info sorted by L* descending.

    Returns:
        List of dicts with keys: bgr (tuple), lab (tuple), label str.
    """
    lab = _bgr_to_lab(pixels_bgr)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, centers = cv2.kmeans(
        lab,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    # Sort centers by L* descending (brightest first)
    sorted_idx = np.argsort(centers[:, 0])[::-1]
    centers_sorted = centers[sorted_idx]

    label_names = ["bright", "dark", "outline"]

    clusters = []
    for i, center_lab in enumerate(centers_sorted):
        name = label_names[i] if i < len(label_names) else f"cluster_{i}"

        # Convert center Lab back to BGR for display
        center_lab_u8 = np.clip(center_lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        center_bgr = cv2.cvtColor(center_lab_u8, cv2.COLOR_Lab2BGR).reshape(3)

        clusters.append(
            {
                "label": name,
                "bgr": (int(center_bgr[0]), int(center_bgr[1]), int(center_bgr[2])),
                "lab": (float(center_lab[0]), float(center_lab[1]), float(center_lab[2])),
            }
        )

    return clusters


def _delta_e76(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """Compute CIE ΔE76 (Euclidean distance in Lab space)."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))


def _bgr_to_lab_single(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert a single BGR uint8 tuple to Lab float."""
    px = np.array([[bgr]], dtype=np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2Lab)
    l_, a, b = lab[0, 0]
    return (float(l_), float(a), float(b))


def _build_collage(ref_path: Path, out_dir: Path) -> Path:
    """Apply kizuato green style and save a 2-column comparison collage.

    Left column: original image.
    Right column: styled image (green, edge-overlay).

    Returns:
        Path to the saved collage PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    styled_path = out_dir / f"{ref_path.stem}_styled_tmp.png"

    apply_kizuato_style(
        ref_path,
        styled_path,
        tint="green",
        outline_strategy="edge-overlay",
    )

    orig = cv2.imread(str(ref_path))
    styled = cv2.imread(str(styled_path))

    # Resize styled to match original height if needed
    if orig.shape[0] != styled.shape[0]:
        scale = orig.shape[0] / styled.shape[0]
        styled = cv2.resize(
            styled,
            (int(styled.shape[1] * scale), orig.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    collage = np.hstack([orig, styled])

    # Limit width to 4000px
    if collage.shape[1] > 4000:
        ratio = 4000 / collage.shape[1]
        collage = cv2.resize(
            collage,
            (4000, int(collage.shape[0] * ratio)),
            interpolation=cv2.INTER_AREA,
        )

    collage_path = out_dir / f"{ref_path.stem}_calibration_collage.png"
    cv2.imwrite(str(collage_path), collage)

    # Remove temp styled image
    styled_path.unlink(missing_ok=True)

    return collage_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate tint palettes against reference scenes."
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=REPO_ROOT / "input" / "kizuato",
        help="Directory containing reference images (default: input/kizuato)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "tools" / "calibration_out",
        help="Directory for output collages (default: tools/calibration_out)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of k-means clusters (default: 3)",
    )
    args = parser.parse_args()

    ref_dir: Path = args.ref_dir
    out_dir: Path = args.out_dir
    k: int = args.k

    if not ref_dir.is_dir():
        print(f"ERROR: ref-dir not found: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    images = _load_images(ref_dir)
    if not images:
        print(f"ERROR: No supported images in {ref_dir}", file=sys.stderr)
        sys.exit(1)

    # Reference: green preset bright/dark/outline in Lab
    green_preset = THREE_TONE_PRESETS["green"]
    green_labels = ["bright", "dark", "outline"]
    green_lab = [_bgr_to_lab_single(bgr) for bgr in green_preset]

    print("=" * 72)
    print("Reference preset 'green' (BGR → Lab):")
    for lbl, bgr, lab in zip(green_labels, green_preset, green_lab):
        print(f"  {lbl:8s}  BGR={bgr}  Lab=({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f})")
    print()

    for ref_path, img in images:
        print("-" * 72)
        print(f"Image: {ref_path.name}  ({img.shape[1]}x{img.shape[0]})")

        pixels = _sample_edge_pixels(img)
        print(f"  Sampled {len(pixels)} pixels (left+right 10% columns)")

        clusters = _kmeans_lab(pixels, k)

        print(f"  k-means k={k} clusters (sorted by L* desc):")
        for cl in clusters:
            lbl = cl["label"]
            bgr = cl["bgr"]
            lab = cl["lab"]

            # Find matching green preset entry for ΔE
            try:
                ref_lab = green_lab[green_labels.index(lbl)]
                de = _delta_e76(lab, ref_lab)
                de_str = f"  ΔE76 vs green.{lbl}={de:.2f}"
            except ValueError:
                de_str = ""

            print(
                f"    {lbl:8s}  BGR={bgr}  Lab=({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}){de_str}"
            )

        # Build comparison collage
        collage_path = _build_collage(ref_path, out_dir)
        print(f"  Collage saved: {collage_path.relative_to(REPO_ROOT)}")
        print()

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
