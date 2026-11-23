# Python Reference Implementation

The PyTorch implementation of the pipeline. It is the reference for the [WebGPU version](../web/), whose shaders are checked against it.

## Requirements

- Python 3.10+
- `torch`, `torchvision`
- `numpy`, `scipy`
- `matplotlib`, `pillow`

```
pip install torch torchvision numpy scipy matplotlib pillow
```

A CUDA GPU is used automatically if available; everything also runs on the CPU.

## Quick Start

Open [`example.ipynb`](example.ipynb) and run all cells (with `python/` as the working directory, the default in Jupyter and VS Code). It processes [`../images/face.jpg`](../images/face.jpg) step by step and shows the intermediate results.

## Pipeline

| Step | Class / Function | What It Does |
|---|---|---|
| Viewing Condition | `ViewConfig` | Display pixel pitch, viewer position and gaze point → per pixel: eccentricity (`ecc_map`), pyramid level (`pyr_level_map`) and equivalent blur sigma (`kernel_map`). `update()` changes gaze or viewer and recomputes the maps. |
| Image | `Image` | Loads the image (8- or 16-bit, colour or grayscale), converts it to YCbCr and foveates it with a per-pixel Gaussian blur (`img.fov`). |
| Pyramids & Masks | `Worm` | Gaussian and Laplacian pyramids, masks/rings per level, cutoff level. |
| Analysis | `Worm.compute_*` | Steerable filters → dominant orientation, phase, intensity per level; intensity modulators from a linear fit of the Laplacian std. |
| Repair | `Worm.repair_*` | Extrapolates orientation, phase and intensity into the regions where a level is blurred away, using the next coarser level. |
| Synthesis | `NoiseGenerator` | Generates one enhancement band per level from the repaired statistics; `get_noise_enhancement` combines them with the masks. |
| Composition | `isu.compose_rgb` | Adds the enhancement signal to the luma of the foveated image → RGB. |

## Experiments

Part 3 of [`example.ipynb`](example.ipynb) renders one example image per experiment; loop over the parameters to produce more. Experiments 1 and 2 can also be explored interactively in the viewer on the [project website](https://sophie-kergassner.github.io/phase-aligned-metamers.html).

1. **Statistic Fidelity**: blends intensity, direction and phase between the original and a fixed random field (`NoiseGenerator.generate_noise(..., keep_intensity, keep_dir, keep_phase, seed)`), with and without repair.
2. **Uniform Foveation**: one pyramid level for the whole image (`ViewConfig(..., uniform_level=...)`).
3. **Statistic Maps per Level**: repaired phase and direction, enhancement band and mask of every synthesized level. To save them as PNGs, use `isu.save_png`.

## Conventions

- **Pyramid**: 5×5 Gaussian (σ = 1), then every 2nd pixel (`[::2]`), no gamma (`isu.get_gauss_mipmaps`).
- **Upsampling** of pyramid levels uses `isu.upsample`. It interpolates at the true sample positions (pixel `2^L · j` for level L). `F.interpolate` would shift level L by `(2^L − 1)/2` px.
- **Blur per Level** (`kernel_map`): the Gaussian sigma that matches "pyramid level L, upsampled bicubically", computed from the pipeline's point spread function (`isu.pyramid_level_to_sigma`). Level 0 means no blur.
- **Eccentricity → Level**: piecewise linear, 0 / 2 / 3 / 4 at 4° / 8° / 16° / 24° (`ViewConfig.map_eccentricity_to_pyr_level`).
- **Post-project fixes**: `Worm(align_phase_convention=...)` switches [fix 1](../fixes/1_phase_sign_alignment.md) (on by default; off reproduces the paper). See [`../fixes/`](../fixes/).

## Package Layout

```
phase_aligned_metamers/
    view_config.py           ViewConfig: viewing geometry and per-pixel maps
    image.py                 Image: loading, YCbCr, per-pixel foveation
    worm.py                  Worm: pyramids, masks, analysis, repair
    noise_generator.py       NoiseGenerator: enhancement signal synthesis
    phase_analysis_utils.py  steerable filters, orientation, phase extrapolation
    imagestats_utils.py      loading, colour conversion, pyramid, upsampling, plotting, saving
example.ipynb                walkthrough and experiments
```
