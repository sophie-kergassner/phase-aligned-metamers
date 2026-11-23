# Fix 1: Phase Sign Alignment at ±90° (post-project fix)

(Earlier name: "Phase Convention Alignment"; the Python switch keeps the name `align_phase_convention`.)

> **Note:** This fix was found **after the project was finished**. The results in the paper do **not** include it.
> The problem was diagnosed with the help of an AI assistant, and the fix was written by Claude (Anthropic).
> It exists in the Python package (`Worm(align_phase_convention=True)`, the default) and in the WebGPU demo (`web/`);
> `align_phase_convention=False` reproduces the paper.

## The symptom

The synthesized noise showed broken, jagged structures along **horizontal** image features: edges and lines broke
into fragments instead of continuing as smooth bands. These were the places where the measured orientation wraps
from −90° to +90°.

## The cause

The orientation `dir = atan2(C3, C2) / 2` is only defined **modulo π**: −89° and +89° are almost the same orientation.
The two steered filters behave differently when the orientation is turned by π:

- G (2nd derivative of a Gaussian) is **even**: `G(θ + π) = G(θ)`
- H (its Hilbert transform) is **odd**: `H(θ + π) = −H(θ)`

So the phase `φ = atan2(H, G)` changes sign. The pairs **(θ, φ)** and **(θ + π, −φ)** describe the same local
signal. The phase is conjugated wherever the orientation crosses ±90°.

The synthesis does not care about this: `cos(φ)·G(θ) − sin(φ)·H(θ)` gives the same result for both pairs. The
**repair** does, because it combines phases of different pixels and levels without checking their convention:

1. **Interpolation across the wrap.** Bicubic upsampling of the complex response `|G + iH|·e^{iφ}` averages `z` with
   `conj(z)`. The imaginary parts cancel, the magnitude drops, and the phase snaps towards 0 or π, which shows up as a
   line with the wrong polarity.
2. **Combining levels with different conventions.** The phase extrapolation `φ = ĉ · conj(f)` with `ĉ = c²/|c|` mixes
   the coarse level `c` and the fine level `f`. If one has an orientation of +89° and the other −89°, one of them is
   conjugated, and both φ and the extrapolated phase are wrong there.
3. **Orientation blend in the single-angle domain.** Where the measured and the extrapolated orientation are blended,
   they were blended as unit vectors `e^{iθ}`. Then +89° and −89° (nearly the same orientation) cancel instead of
   averaging. This blend has to happen in the doubled-angle domain `e^{2iθ}`.

## The fix

Before complex responses with different orientations are interpolated, multiplied or blended, each one is brought
into the convention of a **reference orientation** θ_ref: it is conjugated (its phase negated) wherever its own
orientation points the other way.

```
align_sign(θ, θ_ref) = −1 if cos(θ − θ_ref) < 0, else +1
φ_aligned = align_sign(θ, θ_ref) · φ
```

It is applied in the following places (all in `web/shaders/`):

| Shader | What is aligned | Reference orientation |
|---|---|---|
| `repair_common.wgsl` | `align_sign()`; each of the 16 bicubic samples in `bicubic_mag_phase()` | passed in by the caller |
| `repair_phi.wgsl` | the upsampled coarse level `c_up` before `ĉ · conj(f)` | orientation of `f` (`dir_rep` of level s+1) |
| `repair_level.wgsl` | the upsampled `f` and φ, and the measured phase of level s | the **output** orientation of level s |
| `semantics.wgsl` | the upsampled coarse level (Line / Edge view, display only) | orientation of level l |

In `repair_level.wgsl`, the output orientation is now computed first (blended in the doubled-angle domain, point 3
above), and the output phase is expressed in its convention. That way orientation and phase always flip
**together**, which is exactly what the synthesis expects.

Where the mask keeps the measured statistics (mask = 1), nothing changes. The fix only affects the extrapolated
regions and the transition zones.

## Using it in Python

```python
maps = Worm(device, ycbcr, pyr_level_map, MAX_DEPTH, align_phase_convention=True)   # default; False = as in the paper
```

It changes `Worm.repair_orientation` (doubled-angle blend) and `Worm.repair_phase`. The helpers are in
`phase_analysis_utils.py`: `align_sign`, `upsample_aligned` (bicubic upsampling exactly as `isu.upsample`, but with
each of the 16 taps aligned to the target pixel's orientation, which `grid_sample` cannot do),
`get_phi_quick_aligned` and `extrapolate_phi_aligned`.

## Using it in the demo

- The fix is on by default. The small **"Fix 1 · Phase Sign Alignment: On / Off"** button under the viewer switches it at
  runtime; the button is hidden unless the URL contains `?fix` (e.g. `index.html?fix`).
- `?align=0` in the URL starts with the fix switched off (and also shows the button).
- Internally it is the `align` flag in the repair uniform (`RepairParams`, third value); `0` is the original behaviour.

## Validation

- `web/test_webgpu.py` compares GPU and Python with the fix **on** (default) and **off** (`--no-align`); both match
  (also with `--analyse-foveated`, `--uniform=2.5` and on `mossytree.jpg`). With the fix off, the pipeline is the
  original one.
- With the fix **on**, the noise for `images/face.jpg` was rendered headless and compared visually: the fragmented
  structures around the eyes, nose and mouth (mostly horizontal features) become continuous bands; vertical structures
  are unchanged.

## Status

- WebGPU demo (`web/`): implemented, on by default (2026-11-22).
- Python package: implemented, on by default (`align_phase_convention`, 2026-11-22).
