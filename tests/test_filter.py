from pathlib import Path

import cv2
import numpy as np

from oniazusa.filter import PRESETS, _smooth_tonal_gradient, apply_kizuato_style


def _write_image(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), img)
    assert ok


def test_apply_kizuato_style_preserves_dimensions(tmp_path: Path) -> None:
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[:, :24] = (40, 80, 180)
    img[:, 24:] = (120, 200, 80)

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_kizuato_style(input_path, output_path)
    result = cv2.imread(str(output_path))

    assert result is not None
    assert result.shape == img.shape


def test_smooth_tonal_gradient_softens_local_contrast() -> None:
    gray = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    smoothed = _smooth_tonal_gradient(gray, pre_blur_sigma=1.2, glow_strength=0.2)

    assert 0.0 < float(smoothed[1, 1]) < 1.0


def test_apply_kizuato_style_accepts_all_presets(tmp_path: Path) -> None:
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[:, :] = (60, 120, 220)

    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    for tint in PRESETS:
        output_path = tmp_path / f"{tint}.png"
        apply_kizuato_style(input_path, output_path, tint=tint)
        result = cv2.imread(str(output_path))

        assert result is not None
        assert int(result.sum()) > 0
