# WebGPU Version

Real-time version of the pipeline in the browser. The Python package is the reference: every shader is checked against it by `test_webgpu.py`.

## Status

| Milestone | Content | Status |
|---|---|---|
| 1 | Viewing geometry (eccentricity → level → blur σ), per-pixel foveation, interactive gaze | done, matches Python |
| 2 | Gaussian/Laplacian pyramid, foveation from the pyramid, analysis (orientation, phase, magnitude), intensity fit | done, matches Python |
| 3 | Repair (extrapolation of intensity, orientation and phase from coarser levels) | done, matches Python |
| 4 | Noise synthesis (impulse noise per level), normalization, composition | done, matches Python |

## Running it

WebGPU needs a current Chrome, Edge or Safari (Firefox: Windows). The page has to be served over HTTP; opening `index.html` as a file does not work. From the repository root:

```
python -m http.server 8000
```

Then open <http://localhost:8000/web/>. Move the mouse over the image to move the gaze; click to freeze or release it.

## Deploying to the website

The project page (`phase-aligned-metamers.html` in the website repository) contains the demo's markup and loads `projects/phase-aligned-metamers/realtime/main.js`. After changing anything in `web/`, copy it there with

```
python web/deploy_to_website.py <path to the website repository>
```

It copies `main.js`, `constants.json`, `shaders/` and the six study scenes (`images/*.jpg`). `main.js` is the same file for this development page and the website; the element with `data-pam-demo` sets `data-base`, the folder of the shaders and constants.

## Testing against the Python package

```
pip install wgpu
python web/test_webgpu.py [path/to/image] [--uniform=LEVEL] [--analyse-foveated]
```

The test runs the same `.wgsl` files headless on the local GPU, using the same textures, layouts and dispatch order as `main.js`, and compares every stage with the package: YCbCr, eccentricity, level, σ, both foveations, displayed RGB, every pyramid level, Laplacian band and its std, orientation / phase / magnitude per level, cutoff and intensity fit, the repaired statistics, every noise band and its std, the noise enhancement and the final image. `--uniform=LEVEL` repeats it with one pyramid level for the whole image (other cutoff); `--analyse-foveated` checks the mode in which pyramid and analysis are computed from the foveated image (reference: `Worm(img.fov)`, fed with the GPU's own foveated image so that both analyse the same input; the effect of the float16 blur stack on the noise is printed separately, about 2·10⁻⁵).

Tolerances: float32 rounding differs between GPU and CPU. The phase extrapolation doubles the angle per level, so these differences double from level to level (about 2·10⁻³ rad at L0); a few isolated pixels, where Python's `upscale_complex_phase` multiplies by the sign of an angle close to ±π, can flip. The repair and noise checks therefore require 99.9 % of the pixels to be within tolerance and print the maximum and the number of outliers.

`constants.json` holds the numbers the browser takes from the package (σ calibration, pyramid kernel, steerable filters). The test checks that they still match; after changing them in the package, update the file with `--write-constants`.

## Files

```
index.html                     page and controls
main.js                        WebGPU host: textures, pipelines, per-frame update, UI
shaders/common.wgsl            shared parameters, eccentricity, level and sigma mapping
shaders/sampling.wgsl          torch 'reflect' indexing, shift-free bicubic upsampling (as isu.upsample)
shaders/ycbcr.wgsl             RGB -> YCbCr
shaders/blur.wgsl              separable Gaussian steps of the foveation blur stack (scipy semantics)
shaders/foveate.wgsl           per frame: maps and per-pixel foveation
shaders/foveate_pyramid.wgsl   per frame: foveation by blending upsampled pyramid levels
shaders/pyramid_down.wgsl      one Gaussian pyramid level (5x5, [::2])
shaders/laplace.wgsl           Laplacian band of one level
shaders/analysis_filters.wgsl  7 steerable basis responses and C2, C3
shaders/analysis_orient.wgsl   orientation, steered G/H, phase, magnitude
shaders/repair_common.wgsl     shared repair helpers (complex numbers, masks)
shaders/repair_init.wgsl       levels >= cutoff: measured statistics
shaders/repair_phi.wgsl        phase repair, step 1 (phi = c_hat * conj(f))
shaders/repair_level.wgsl      repair of one level (intensity, orientation, phase)
shaders/noise_band.wgsl        noise band of one level (64 impulse grid positions)
shaders/noise_stats.wgsl       std of the upsampled band, step 1 (partial sums)
shaders/noise_finalize.wgsl    std, step 2
shaders/lap_stats.wgsl         band std for the fit, step 1 (masked partial sums)
shaders/lap_finalize.wgsl      band std and weight, step 2
shaders/fit.wgsl               intensity fit (weighted or paper) on the GPU
shaders/compose.wgsl           noise enhancement and final image
shaders/display.wgsl           view modes and gaze marker
constants.json                 numbers from the package (sigma calibration, kernels)
test_webgpu.py                 headless comparison with the Python package
deploy_to_website.py           copies the demo into the website repository
```

## Notes

- Everything that does not depend on the gaze (blur stack, pyramid, Laplacian bands and their std, analysis) is computed once per image. Per frame: the maps, both foveations, cutoff and intensity fit (on the CPU, from the stored std values), repair, noise bands, their std and the final image.
- "Analyse: original" measures the statistics on the original image, as the Python package does; the masks decide where they are kept and where they are repaired. "Analyse: foveated" builds pyramid and analysis from the foveated base image every frame, as a renderer that only has the foveated image would; there, the statistics of blurred-away levels are visibly degenerate.
- Intensity fit: the web version uses a **weighted fit** by default (not in the paper, and not in the Python package): a line through the std of all Laplacian bands, each measured only where its level is fully kept (mask = 1; the soft transition is left out as unreliable) and weighted by that fraction of the image. Unlike the paper's fit (bands from the cutoff level on, still selectable as "paper"), it does not jump when the cutoff level changes (e.g. around 125 cm viewing distance). The plot under the demo shows the band stds, their weights, the line and the resulting noise amplitudes. `test_webgpu.py` checks both (`--fit=cutoff` for the paper fit).
- The Laplacian std and the intensity fit run on the GPU, so they follow the per-frame analysis; the values are read back asynchronously for the info line and the legends.
- Viewing setup: the viewer measures the image's width on the screen with a ruler and enters it; that fixes the size of one device pixel (remembered in localStorage), so the physical width (shown CSS width x devicePixelRatio x cm per device px) follows window size, browser / OS zoom and full screen (full-screen button in the corner of the image, Esc to leave).
- Effect of the parameters (noise and result stages): sliders for the share of the measured intensity / orientation / phase used in the noise (NoiseGenerator keep_*; 1 = as measured, 0 = random). The random fields come from a hash (fixed while sliding); `test_webgpu.py --keep=I,D,P` checks it with the same fields.
- Statistics views: "Measured" and "Repaired" (after the repair). "Measure statistics on": the foveated image (default, what a renderer has) or the full-res original (as in the paper and the Python package). They differ mainly in the intensity: re-analysing the foveated image builds the Gaussian pyramid a second time, so bands near the blurred levels lose some more frequencies.
- While the mouse is outside the image, the gaze follows a slow automatic path (switchable; off = image centre). A click on the image freezes the gaze.
- Two foveations are available: per-pixel blur (`img.fov` in Python, the base image of the pipeline) and blending of the upsampled pyramid levels (`Worm.compute_foveated_imgs`). The view "Difference blur − pyramid" compares them.
- Stack layers are stored as 16-bit floats (176 MB at 2048 × 1024 instead of 352 MB). This changes the foveated image by at most 5·10⁻⁴, about an eighth of an 8-bit step; the fovea is exact.
