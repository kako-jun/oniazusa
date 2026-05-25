import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from oniazusa.filter import PRESETS
from oniazusa.cli import main


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

    with patch.object(sys, "argv", ["oniazusa", str(in_dir), "-o", str(out_dir), "--mode", "three-tone"]):
        main()

    assert (out_dir / "a_three_tone.png").exists()
    assert (out_dir / "b_three_tone.png").exists()


def test_cli_mode_three_tone_ignores_levels(tmp_path: Path) -> None:
    img = np.full((24, 24, 3), 200, dtype=np.uint8)
    input_path = tmp_path / "photo.png"
    _write_image(input_path, img)

    with patch.object(sys, "argv", ["oniazusa", str(input_path), "--mode", "three-tone", "--levels", "8"]):
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
