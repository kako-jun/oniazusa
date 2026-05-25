# oniazusa Overview

`oniazusa` is a mood-first image filter for turning photos into Kizuato-like visual-novel
backgrounds.

## Goal

Reduce photographic realism and replace it with a low-resolution, tinted, dithered look that
feels like a remembered scene rather than a direct capture.

## Intended use cases

- outdoor backgrounds for `skirts-colour`
- testing tint direction before production painting
- checking whether a photo survives reduction into a VN-style still
- acting as a reusable background-processing step from other tools

## Non-goals

- generic anime filter marketing
- portrait beautification
- physically accurate segmentation or relighting

## Position in the broader pipeline

- `oniazusa` transforms the whole scene mood
- `patcolour` can selectively preserve color where needed
- `skirts-colour` is the production context where both experiments matter
- `name-name` and other tools may compose this filter inside larger asset pipelines
