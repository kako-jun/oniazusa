from pathlib import Path

import cv2
import numpy as np
import pytest

from oniazusa.filter import (
    OUTLINE_STRATEGIES,
    PRESETS,
    THREE_TONE_PRESETS,
    _detect_edge_map,
    _smooth_tonal_gradient,
    apply_comparison,
    apply_kizuato_style,
    apply_three_tone,
)


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


def test_apply_three_tone_pixel_dark_zone(tmp_path: Path) -> None:
    # 中間グレー画像(gray≈0.50、t1<x<t2) → dark_bgr に揃う
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    input_path = tmp_path / "mid.png"
    output_path = tmp_path / "mid_out.png"
    _write_image(input_path, img)

    apply_three_tone(input_path, output_path, tint="green")
    result = cv2.imread(str(output_path))

    assert result is not None
    dark_bgr = THREE_TONE_PRESETS["green"][1]
    bright_bgr = THREE_TONE_PRESETS["green"][0]
    outline_bgr = THREE_TONE_PRESETS["green"][2]
    # dark_bgr が最多数のピクセルを占める（前処理で一様画像は中間帯に収まる）
    n_dark = int(
        np.sum(
            (result[:, :, 0] == dark_bgr[0])
            & (result[:, :, 1] == dark_bgr[1])
            & (result[:, :, 2] == dark_bgr[2])
        )
    )
    n_bright = int(
        np.sum(
            (result[:, :, 0] == bright_bgr[0])
            & (result[:, :, 1] == bright_bgr[1])
            & (result[:, :, 2] == bright_bgr[2])
        )
    )
    n_outline = int(
        np.sum(
            (result[:, :, 0] == outline_bgr[0])
            & (result[:, :, 1] == outline_bgr[1])
            & (result[:, :, 2] == outline_bgr[2])
        )
    )
    assert n_dark >= n_bright and n_dark >= n_outline


def _make_edge_image() -> np.ndarray:
    """100x100 の黒背景に白矩形を描いた画像（明確なエッジあり）。"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    return img


# ---------------------------------------------------------------------------
# _detect_edge_map
# ---------------------------------------------------------------------------


def test_detect_edge_map_returns_float32() -> None:
    img = _make_edge_image()
    edge_map = _detect_edge_map(img)
    assert edge_map.dtype == np.float32
    assert edge_map.shape == img.shape[:2]


def test_detect_edge_map_range() -> None:
    img = _make_edge_image()
    edge_map = _detect_edge_map(img)
    assert float(edge_map.min()) >= 0.0
    assert float(edge_map.max()) <= 1.0


def test_detect_edge_map_detects_edges() -> None:
    img = _make_edge_image()
    edge_map = _detect_edge_map(img)
    assert float(edge_map.max()) == 1.0


# ---------------------------------------------------------------------------
# apply_kizuato_style — outline_strategy
# ---------------------------------------------------------------------------


def test_apply_kizuato_style_default_strategy_is_edge_overlay(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_kizuato_style(input_path, output_path)
    assert output_path.exists()


def test_apply_kizuato_style_edge_overlay_matches_default(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    out_default = tmp_path / "default.png"
    out_explicit = tmp_path / "explicit.png"
    apply_kizuato_style(input_path, out_default)
    apply_kizuato_style(input_path, out_explicit, outline_strategy="edge-overlay")

    result_default = cv2.imread(str(out_default))
    result_explicit = cv2.imread(str(out_explicit))
    assert result_default is not None
    assert result_explicit is not None
    assert np.array_equal(result_default, result_explicit)


def test_apply_kizuato_style_all_strategies_produce_output(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    for strategy in OUTLINE_STRATEGIES:
        out = tmp_path / f"{strategy}.png"
        apply_kizuato_style(input_path, out, outline_strategy=strategy)
        assert out.exists(), f"{strategy} did not produce output"


def test_apply_kizuato_style_strategies_differ_from_each_other(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    results = {}
    for strategy in OUTLINE_STRATEGIES:
        out = tmp_path / f"{strategy}.png"
        apply_kizuato_style(input_path, out, outline_strategy=strategy)
        results[strategy] = cv2.imread(str(out))

    strategies = list(results.keys())
    all_same = all(
        np.array_equal(results[strategies[0]], results[s]) for s in strategies[1:]
    )
    assert not all_same, "All strategies produced identical output"


# ---------------------------------------------------------------------------
# apply_comparison
# ---------------------------------------------------------------------------


def test_apply_comparison_returns_five_paths(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_comparison(input_path, tmp_path / "out")
    assert len(paths) == 5


def test_apply_comparison_all_files_exist(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_comparison(input_path, tmp_path / "out")
    for p in paths:
        assert p.exists(), f"{p} does not exist"


def test_apply_comparison_individual_names_contain_strategy(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_comparison(input_path, tmp_path / "out")
    individual = paths[:4]
    for strategy, path in zip(OUTLINE_STRATEGIES, individual):
        assert strategy in path.name, f"{strategy} not in {path.name}"


def test_apply_comparison_collage_name(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_comparison(input_path, tmp_path / "out")
    collage = paths[-1]
    assert collage.name == "input_compare.png"


def test_apply_comparison_collage_width_le_4000(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_comparison(input_path, tmp_path / "out")
    collage = cv2.imread(str(paths[-1]))
    assert collage is not None
    assert collage.shape[1] <= 4000


def test_apply_comparison_collage_is_wider_than_individual(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_comparison(input_path, tmp_path / "out")
    individual = cv2.imread(str(paths[0]))
    collage = cv2.imread(str(paths[-1]))
    assert individual is not None
    assert collage is not None
    assert collage.shape[1] > individual.shape[1]
