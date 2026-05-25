from pathlib import Path

import cv2
import numpy as np
import pytest

from oniazusa.filter import PRESETS, THREE_TONE_PRESETS, _smooth_tonal_gradient, apply_kizuato_style, apply_three_tone


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


def test_apply_three_tone_preserves_dimensions(tmp_path: Path) -> None:
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[:, :24] = (40, 80, 180)
    img[:, 24:] = (120, 200, 80)

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_three_tone(input_path, output_path)
    result = cv2.imread(str(output_path))

    assert result is not None
    assert result.shape == img.shape


def test_apply_three_tone_accepts_all_presets(tmp_path: Path) -> None:
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[:, :] = (60, 120, 220)

    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    for tint in ("green", "yellow", "blue", "purple"):
        output_path = tmp_path / f"{tint}.png"
        apply_three_tone(input_path, output_path, tint=tint)
        result = cv2.imread(str(output_path))

        assert result is not None
        assert int(result.sum()) > 0


def test_apply_three_tone_pixel_bright_zone(tmp_path: Path) -> None:
    # 全白画像(gray≈1.0 > t2=0.72) → bright_bgr に揃う
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    input_path = tmp_path / "white.png"
    output_path = tmp_path / "white_out.png"
    _write_image(input_path, img)

    apply_three_tone(input_path, output_path, tint="green")
    result = cv2.imread(str(output_path))

    assert result is not None
    bright_bgr = THREE_TONE_PRESETS["green"][0]
    # 全ピクセルが bright_bgr と一致することを確認
    assert np.all(result[:, :, 0] == bright_bgr[0])
    assert np.all(result[:, :, 1] == bright_bgr[1])
    assert np.all(result[:, :, 2] == bright_bgr[2])


def test_apply_three_tone_pixel_outline_zone(tmp_path: Path) -> None:
    # 全黒画像(gray≈0.0 < t1=0.45) → outline_bgr に揃う
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    input_path = tmp_path / "black.png"
    output_path = tmp_path / "black_out.png"
    _write_image(input_path, img)

    apply_three_tone(input_path, output_path, tint="green")
    result = cv2.imread(str(output_path))

    assert result is not None
    outline_bgr = THREE_TONE_PRESETS["green"][2]
    assert np.all(result[:, :, 0] == outline_bgr[0])
    assert np.all(result[:, :, 1] == outline_bgr[1])
    assert np.all(result[:, :, 2] == outline_bgr[2])


def test_apply_three_tone_unknown_tint_falls_back_to_green(tmp_path: Path) -> None:
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[:, :] = (60, 120, 220)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # nonexistent tint → green フォールバックで正常終了
    apply_three_tone(input_path, output_path, tint="nonexistent")
    result = cv2.imread(str(output_path))

    assert result is not None
    assert result.shape[:2] == (24, 24)


def test_apply_three_tone_invalid_input_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        apply_three_tone(tmp_path / "no_such_file.png", tmp_path / "out.png")


def test_three_tone_presets_keys_match_presets() -> None:
    assert set(THREE_TONE_PRESETS.keys()) == set(PRESETS.keys())


def test_apply_three_tone_output_is_three_channel(tmp_path: Path) -> None:
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[:, :] = (100, 150, 200)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_three_tone(input_path, output_path)
    result = cv2.imread(str(output_path))

    assert result is not None
    assert result.ndim == 3
    assert result.shape[2] == 3
