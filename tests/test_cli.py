from oniazusa.filter import PRESETS


def test_presets_are_available() -> None:
    assert {"green", "yellow", "blue", "purple"} <= set(PRESETS)
