# Post-Project Fixes

Corrections found **after the project was finished**. The results in the paper do **not** include them. Each fix can
be switched off to reproduce the paper's behaviour.

| Fix | Problem | Switch (Python / web demo) | Status |
|---|---|---|---|
| 1 · [Phase Sign Alignment at ±90°](1_phase_sign_alignment.md) | Broken structures in the noise along horizontal features: the phase flips sign where the orientation wraps at ±90°, and the repair mixed both conventions. The phases are now aligned to one convention before they are interpolated or combined across levels. | `Worm(align_phase_convention=...)` / `?fix` (button), `?align=0` | Python + web, on by default (2026-11-22) |
| 2 · [Orientation Smoothing](2_orientation_smoothing.md) | The orientation is unstable: `atan2(C3, C2)` turns by large angles for tiny changes of the input where C2, C3 are near zero (a 1-px shift of the image changes it noticeably), and the steered phase follows it. C2, C3 are now averaged over a 7×7 Gaussian window (σ = 2 px) instead of the 5×5 box, and the repair weights orientations by their confidence. | – / `?fix` (button), `?orient=0` | web only, on by default (2026-11-22) |
