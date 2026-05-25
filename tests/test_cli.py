import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from oniazusa.cli import main
from oniazusa.filter import PREPROCESS_MODES, PRESETS


def _write_image(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), img)
    assert ok


def test_presets_are_available() -> None:
    assert {"green", "yellow", "blue", "purple"} <= set(PRESETS)


def test_cli_mode_three_tone_single_file_default_output_name(tmp_path: Path) -> None:
    img = np.full((24, 24, 3), 200, dtype=np.uint8)
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    with patch.object(sys, "argv", ["oniazusa", str(input_path), "--mode", "three-tone"]):
        main()

    expected = tmp_path / "photo_three_tone.png"
    assert expected.exists()


def test_cli_mode_kizuato_single_file_default_output_name(tmp_path: Path) -> None:
    img = np.full((24, 24, 3), 200, dtype=np.uint8)
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    with patch.object(sys, "argv", ["oniazusa", str(input_path), "--mode", "kizuato"]):
        main()

    expected = tmp_path / "photo_kizuato.png"
    assert expected.exists()


def test_cli_mode_three_tone_directory_output_names(tmp_path: Path) -> None:
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    for name in ("a.png", "b.png"):
        img = np.full((24, 24, 3), 200, dtype=np.uint8)
        _write_image(in_dir / name, img)

    out_dir = tmp_path / "out"

    argv = ["oniazusa", str(in_dir), "-o", str(out_dir), "--mode", "three-tone"]
    with patch.object(sys, "argv", argv):
        main()

    assert (out_dir / "a_three_tone.png").exists()
    assert (out_dir / "b_three_tone.png").exists()


def test_cli_mode_three_tone_ignores_levels(tmp_path: Path) -> None:
    img = np.full((24, 24, 3), 200, dtype=np.uint8)
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    argv = ["oniazusa", str(input_path), "--mode", "three-tone", "--levels", "8"]
    with patch.object(sys, "argv", argv):
        main()

    expected = tmp_path / "photo_three_tone.png"
    assert expected.exists()


def test_cli_invalid_mode_rejected(tmp_path: Path) -> None:
    img = np.full((24, 24, 3), 200, dtype=np.uint8)
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    with patch.object(sys, "argv", ["oniazusa", str(input_path), "--mode", "invalid_mode"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 2


def test_cli_mode_kizuato_directory_output_names(tmp_path: Path) -> None:
    # kizuato ディレクトリモードでも出力が {stem}_kizuato.png になるリグレッション確認
    in_dir = tmp_path / "imgs"
    in_dir.mkdir()
    for name in ["a.png", "b.png"]:
        img = np.full((24, 24, 3), 200, dtype=np.uint8)
        _write_image(in_dir / name, img)

    out_dir = tmp_path / "out_kizuato"
    argv = ["oniazusa", str(in_dir), "-o", str(out_dir), "--mode", "kizuato"]
    with patch.object(sys, "argv", argv):
        main()

    assert (out_dir / "a_kizuato.png").exists()
    assert (out_dir / "b_kizuato.png").exists()


# ---------------------------------------------------------------------------
# outline_strategy / compare CLI tests
# ---------------------------------------------------------------------------


def _make_edge_image() -> np.ndarray:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    return img


def test_cli_outline_strategy_passed_to_apply(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    with patch("oniazusa.cli.apply_kizuato_style") as mock_apply:
        argv = ["oniazusa", str(input_path), "--outline-strategy", "edge-bias"]
        with patch.object(sys, "argv", argv):
            main()

    mock_apply.assert_called_once()
    _, kwargs = mock_apply.call_args
    assert kwargs.get("outline_strategy") == "edge-bias"


def test_cli_compare_calls_apply_comparison(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    out_dir = tmp_path / "out"
    fake_paths = [out_dir / f"x_{i}.png" for i in range(5)]

    with patch("oniazusa.cli.apply_comparison", return_value=fake_paths) as mock_cmp:
        argv = ["oniazusa", str(input_path), "--compare", "-o", str(out_dir)]
        with patch.object(sys, "argv", argv):
            main()

    mock_cmp.assert_called_once()


def test_cli_compare_overrides_outline_strategy(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    out_dir = tmp_path / "out"
    fake_paths = [out_dir / f"x_{i}.png" for i in range(5)]

    with (
        patch("oniazusa.cli.apply_comparison", return_value=fake_paths) as mock_cmp,
        patch("oniazusa.cli.apply_kizuato_style") as mock_kiz,
    ):
        argv = [
            "oniazusa",
            str(input_path),
            "--compare",
            "--outline-strategy",
            "dither-density",
            "-o",
            str(out_dir),
        ]
        with patch.object(sys, "argv", argv):
            main()

    mock_cmp.assert_called_once()
    mock_kiz.assert_not_called()


# ---------------------------------------------------------------------------
# --preprocess / --compare-preprocess CLI tests
# ---------------------------------------------------------------------------


def test_cli_preprocess_passed_to_apply_kizuato_style(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    with patch("oniazusa.cli.apply_kizuato_style") as mock_apply:
        argv = ["oniazusa", str(input_path), "--preprocess", "flatten"]
        with patch.object(sys, "argv", argv):
            main()

    mock_apply.assert_called_once()
    _, kwargs = mock_apply.call_args
    assert kwargs.get("preprocess") == "flatten"


def test_cli_compare_preprocess_calls_apply_comparison_preprocess(tmp_path: Path) -> None:
    img = _make_edge_image()
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    out_dir = tmp_path / "out"
    fake_paths = [out_dir / f"x_{i}.png" for i in range(5)]

    with patch("oniazusa.cli.apply_comparison_preprocess", return_value=fake_paths) as mock_cp:
        argv = ["oniazusa", str(input_path), "--compare-preprocess", "-o", str(out_dir)]
        with patch.object(sys, "argv", argv):
            main()

    mock_cp.assert_called_once()


def test_cli_compare_preprocess_takes_precedence_over_preprocess(tmp_path: Path) -> None:
    # --compare-preprocess と --preprocess 同時指定では compare_preprocess が優先
    img = _make_edge_image()
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    out_dir = tmp_path / "out"
    fake_paths = [out_dir / f"x_{i}.png" for i in range(5)]

    with (
        patch("oniazusa.cli.apply_comparison_preprocess", return_value=fake_paths) as mock_cp,
        patch("oniazusa.cli.apply_kizuato_style") as mock_kiz,
    ):
        argv = [
            "oniazusa",
            str(input_path),
            "--compare-preprocess",
            "--preprocess",
            "flatten",
            "-o",
            str(out_dir),
        ]
        with patch.object(sys, "argv", argv):
            main()

    mock_cp.assert_called_once()
    mock_kiz.assert_not_called()


def test_cli_compare_preprocess_before_compare_in_directory_mode(tmp_path: Path) -> None:
    # --compare-preprocess と --compare 同時指定では compare_preprocess が先（directory mode）
    in_dir = tmp_path / "imgs"
    in_dir.mkdir()
    img = np.full((24, 24, 3), 200, dtype=np.uint8)
    _write_image(in_dir / "a.png", img)

    out_dir = tmp_path / "out"
    fake_paths = [out_dir / f"x_{i}.png" for i in range(5)]

    with (
        patch("oniazusa.cli.apply_comparison_preprocess", return_value=fake_paths) as mock_cp,
        patch("oniazusa.cli.apply_comparison") as mock_cmp,
    ):
        argv = [
            "oniazusa",
            str(in_dir),
            "--compare-preprocess",
            "--compare",
            "-o",
            str(out_dir),
        ]
        with patch.object(sys, "argv", argv):
            main()

    mock_cp.assert_called()
    mock_cmp.assert_not_called()


def test_cli_preprocess_directory_mode_applies_to_all_files(tmp_path: Path) -> None:
    # ディレクトリ入力 + --compare-preprocess で全ファイルに apply_comparison_preprocess が呼ばれる
    in_dir = tmp_path / "imgs"
    in_dir.mkdir()
    for name in ["a.png", "b.png", "c.png"]:
        img = np.full((24, 24, 3), 200, dtype=np.uint8)
        _write_image(in_dir / name, img)

    out_dir = tmp_path / "out"
    fake_paths = [out_dir / f"x_{i}.png" for i in range(5)]

    with patch("oniazusa.cli.apply_comparison_preprocess", return_value=fake_paths) as mock_cp:
        argv = ["oniazusa", str(in_dir), "--compare-preprocess", "-o", str(out_dir)]
        with patch.object(sys, "argv", argv):
            main()

    assert mock_cp.call_count == 3
