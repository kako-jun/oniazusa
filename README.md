# oniazusa

[![CI](https://github.com/kako-jun/oniazusa/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/oniazusa/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/oniazusa.svg)](https://pypi.org/project/oniazusa/)

Transform photos into visual-novel backgrounds inspired by Leaf's *Kizuato* (`痕`) style.

`oniazusa` is a standalone CLI for testing the mood, tint, pixel structure, and screen-tone
feel of low-resolution, memory-like backgrounds before they are folded into `skirts-colour`.
It is also expected to act as reusable image-processing infrastructure for other tools.

## Why oniazusa?

The target is not "anime filter" in the broad web sense. The tool is trying to capture a
specific late-90s visual-novel background feeling: reduced detail, tinted paper-like light,
and dithered depth instead of photographic realism.

That means the output style matters, but interface stability matters too, because this tool is
meant to be called from `name-name`-based workflows and other repos.

## Install

```bash
uv tool install oniazusa
```

Or from source:

```bash
git clone https://github.com/kako-jun/oniazusa.git
cd oniazusa
uv sync --group dev
uv run oniazusa --help
```

## Usage

### Single image

```bash
oniazusa photo.jpg -o background.png
```

### Batch processing

```bash
oniazusa ./photos/ -o ./backgrounds/
```

### Change tint and dithering levels

```bash
oniazusa photo.jpg --tint green --levels 24 -o background.png
```

### Shape the gradient before the visible grid

```bash
oniazusa photo.jpg --pre-blur 1.8 --glow 0.24 -o background.png
```

## Current behavior

- bilateral filtering to flatten texture while preserving edges
- soft edge overlay from Canny outlines
- Gaussian blur plus glow-like smoothing before ordered grid dithering
- heavy downscale then ordered Bayer dithering
- tint mapping through a limited preset palette
- nearest-neighbor upscale for visible dot structure

The CLI is only the first surface. The underlying behavior should remain stable enough for
other tools to depend on it.

## Limitations

- foreground and background are processed uniformly
- tint presets are still rough and not yet matched against real reference scenes
- `--preprocess` supports `none` (default), `denoise`, `flatten`, `illustration` modes; use `--compare-preprocess` to compare all modes side by side

## Docs

- `docs/overview.md` — artistic target and scope
- `docs/spec.md` — CLI and filter steps
- `docs/roadmap.md` — quality gaps and next work
- `CLAUDE.md` — AI-facing internal guide

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
```
