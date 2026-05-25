# Research Notes

Short notes gathered while refining `oniazusa`.

## PC-98 constraints that matter here

- The classic PC-98 graphics target was commonly `640x400` with `16 colors` chosen from a
  `4096-color` palette.
- That means the "PC-98 look" is not just low color count. It is also deliberate palette
  selection plus structured dithering to fake smoother ramps.

## Practical implication for `oniazusa`

- Avoid treating the target look as generic posterization.
- Prefer a small number of intentional tones, then let ordered dithering carry the transition.
- A 3-tone model is a good default mental model:
  - bright background tone
  - darker background tone
  - strongest dark tone for outlines / tight shadows

## Ordered dithering direction

- The relevant family name is `ordered dithering`, commonly using a Bayer threshold matrix.
- The important sequence is:
  1. smooth the large-scale tonal flow
  2. quantize into a small set of tones
  3. use ordered dithering where transitions need to survive

If step 1 is weak, the result tends to become harsh or dirty instead of elegant.

## Outline treatment

The likely candidates are:

- explicit edge mask gets the darkest tone
- edges are only biased darker inside shadow zones
- edges are suggested by denser dither rather than a separate dark line
- thresholds are shifted locally near edges

This is why the repo keeps outline handling as a comparison problem, not a solved fact.

## Reference-reading rule

When using `input/kizuato`, ignore foreground character stand sprites.
Evaluate the background only, with special attention to left-edge and right-edge atmosphere.

## Additional real-world input set

The committed night backgrounds from `ear-sky/public/bg/` are useful as a second test family.

Why:

- they are not trying to imitate PC-98 directly
- they stress dark scenes, neon contrast, haze, and bokeh
- they reveal whether the reduction pipeline preserves atmosphere or simply crushes it
