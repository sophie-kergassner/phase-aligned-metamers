# Fix 2: Orientation Smoothing (post-project fix)

(Earlier name: "Orientation Stability".)

> **Note:** This fix was found **after the project was finished**. The results in the paper do **not** include it.
> The problem was diagnosed with the help of an AI assistant, and the fix was written by Claude (Anthropic).
> So far it only exists in the WebGPU demo (`web/`); the Python package keeps the paper's orientation estimate.

## The symptom

The measured orientation is very sensitive to small changes of the input: shifting the image by a single pixel
changes it by large angles in places, although the Laplacian bands change only slightly. This is strongest where the
orientation map has steep gradients. The intensity map is hardly affected, but the orientation map and the Line / Edge
map change, and with them the structure of the enhancement.

## The cause

The orientation is estimated per pixel from the two orientation energy terms C2 and C3:

```
dir = atan2(C3, C2) / 2        S = |C2 + i·C3|   (orientation strength)
```

C2 and C3 are averaged over a 5×5 box first (`phu.get_theta_d`). Where `C2 + i·C3` is close to zero (corners,
crossings, between structures of different orientation, weak texture), `atan2` turns by large angles for tiny changes
of its input. Shifting the image by one pixel causes such tiny changes: the Laplacian bands move smoothly, but the angle
taken from them does not.

This error spreads: G and H are **steered** with `dir`, so the measured phase `atan2(H, G)` changes with it, and the
repair upsamples and combines these orientations and phases into the extrapolated levels. The intensity `|G + iH|`
hardly depends on the steering angle, which is why it stays smooth.

A second, smaller cause is in the repair: the orientation of a coarse level is upsampled as the (bicubic) sum of 16
unit vectors `e^{2i·dir}`, which is then normalised. Where the 16 taps disagree, the sum is short and its angle is
arbitrary, but after normalising it counts as much as a reliable orientation.

## The fix

1. **Gaussian averaging window for C2, C3.** C2 and C3 are averaged over a 7×7 Gaussian window (σ = 2 px of each
   level, zero padding, as two separable passes) instead of the 5×5 box. The box corresponds to σ ≈ 1.4 px.
   Averaging in the (C2, C3) domain is the doubled-angle domain, so the ±90° wrap needs no special treatment there.
2. **Confidence-weighted orientation upsampling in the repair.** Each of the 16 taps is weighted by the confidence of
   its orientation: `S` where the level is measured, and the length of the weighted sum where it was extrapolated
   itself. Taps without a clear orientation then no longer decide the direction.

| Shader (`web/shaders/`) | Change |
|---|---|
| `orient_smooth.wgsl` (new) | `smooth_h`, `smooth_v`: Gaussian average of C2, C3 (`ORIENT_SIGMA = 2`, `ORIENT_RADIUS = 3`) |
| `analysis_orient.wgsl` | entry point `main_presmoothed`: orientation from the already averaged C2, C3 (no box) |
| `repair_init.wgsl` | writes the confidence `S` into the 4th channel of the repaired statistics |
| `repair_level.wgsl` | orientation taps weighted by that confidence; writes the confidence of its own output |

## Effect

Change of the measured orientation between the image and the image shifted by 1 px, weighted by `|G + iH|` (measured
headless on the GPU with the demo's shaders, `face.jpg`, 2048 × 1024, level 0):

| | mean change | share of changes > 30° |
|---|---|---|
| 5×5 box (paper) | 6.7° | 3.6 % |
| 5×5 Gaussian, σ = 1 | 9.5° | 6.5 % |
| **7×7 Gaussian, σ = 2 (fix)** | **5.3°** | **2.3 %** |
| 17×17 Gaussian, σ = 3 | 2.9° | 0.7 % |

Level 0 (the finest level, used around the gaze point) is by far the most unstable; levels 1–3 are 3–10× steadier
already without the fix. The wider the window, the steadier the orientation, but the more the orientation map is
smoothed. A first version used 17×17 (σ = 3); it smoothed the orientation map (and therefore the structure of the
noise) too much, so the window was reduced to 7×7. A 5×5 Gaussian is narrower than the 5×5 box and less stable than it.

**Trade-off:** the orientation map becomes somewhat smoother. Fine orientation detail at corners and crossings is
averaged out, which changes the measured statistics compared with the paper.

## Computational cost

The method's complexity does not change: it is still linear in the number of pixels. Per pixel, the 25 taps of the
box (computed in one pass, not separated) are replaced by 2 × 7 Gaussian taps in two separable passes. The repair
change is one multiplication per tap.

Measured on an RTX 4090 for level 0 at 2048 × 1024:

| | time |
|---|---|
| basis filters (9×9 × 7, unchanged) | ≈ 0.8–1.0 ms |
| orientation with 5×5 box | ≈ 0.13 ms |
| 7×7 Gaussian (2 passes) + orientation | ≈ 0.11 ms |

So the fix costs no extra time (within measurement noise); the analysis is dominated by the basis filters. The
17×17 version took ≈ 0.19 ms.

Memory: one extra texture per analysed level for the intermediate result of the separable filter (currently
`rgba32float`, about 45 MB per pyramid set at 2048 × 1024; `rg32float` would halve it).

## Using it in the demo

- The fix is on by default. The small **"Fix 2 · Orientation Smoothing: On / Off"** button under the viewer switches it at runtime;
  the button is hidden unless the URL contains `?fix` (or `?orient=...`).
- `?orient=0` in the URL starts with the fix switched off.
- Internally: `state.orientFix` in `main.js` (selects the analysis passes) and the 4th value of the repair
  uniform (`RepairParams.orient`). Switching it rebuilds the analysis of the original image.

## Validation

- `web/test_webgpu.py` runs with the fix **off** and still matches the Python package.
- The GPU result of `orient_smooth.wgsl` + `main_presmoothed` matches a PyTorch Gaussian average of C2, C3 (99.9 % of
  pixels within 1e-6 rad).
- The visual effect was checked in the browser: the orientation map becomes much smoother.

## Status

- Web demo only (2026-11-22); Python port open, as for the phase alignment it should first prove itself in the demo.
