# Phase-Aligned Metamers (Python & WebGPU)

![Teaser](teaser.jpg)

Phase-aligned enhancement for foveated images. Detail lost through foveation (blur in the periphery) is replaced by a synthesized enhancement signal whose local orientation, phase and intensity follow the image's statistics, extrapolated from coarser scales.

This repository has two implementations of the same pipeline:

- **Python/PyTorch** ([`phase_aligned_metamers/`](phase_aligned_metamers/)): the reference implementation, with a step-by-step walkthrough in [`example.ipynb`](example.ipynb).
- **Real-time WebGPU** ([`web/`](web/)): the full pipeline on the GPU in the browser, updating live as you move your gaze. See [Real-Time WebGPU Version](#real-time-webgpu-version).

**Project website:** [sophie-kergassner.github.io/phase-aligned-metamers](https://sophie-kergassner.github.io/phase-aligned-metamers.html). Try the interactive real-time WebGPU viewer there, and find more information: an explanatory video, the paper and other material.

## Real-Time WebGPU Version

[`web/`](web/) runs the full pipeline in real time in the browser with WebGPU. The exact same code runs on the [project page](https://sophie-kergassner.github.io/phase-aligned-metamers.html), so you can try it there without installing anything. JavaScript ([`main.js`](web/main.js)) sets up the GPU passes. The pipeline steps themselves (foveation, pyramids, analysis, repair, synthesis of the enhancement signal, composition) are compute shaders written in WGSL, the WebGPU Shading Language, in [`web/shaders/`](web/shaders/). The Python package is the reference: [`web/test_webgpu.py`](web/test_webgpu.py) checks every shader against it.

To run it locally, serve the repository root over HTTP (`python -m http.server 8000`) and open <http://localhost:8000/web/> in a current Chrome, Edge or Safari. See [`web/README.md`](web/README.md) for details.

## Requirements

- Python 3.10+
- `torch`, `torchvision`
- `numpy`, `scipy`
- `matplotlib`, `pillow`

A CUDA GPU is used automatically if available; everything also runs on the CPU.

## Quick Start

Open [`example.ipynb`](example.ipynb) and run all cells. It processes `images/face.jpg` step by step and shows the intermediate results.

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

Part 3 of `example.ipynb` renders one example image per experiment; loop over the parameters to produce more. Experiments 1 and 2 can also be explored interactively in the viewer on the project website.

1. **Statistic Fidelity**: blends intensity, direction and phase between the original and a fixed random field (`NoiseGenerator.generate_noise(..., keep_intensity, keep_dir, keep_phase, seed)`), with and without repair.
2. **Uniform Foveation**: one pyramid level for the whole image (`ViewConfig(..., uniform_level=...)`).
3. **Statistic Maps per Level**: repaired phase and direction, enhancement band and mask of every synthesized level. To save them as PNGs, use `isu.save_png`.

## Conventions

- **Pyramid**: 5×5 Gaussian (σ = 1), then every 2nd pixel (`[::2]`), no gamma (`isu.get_gauss_mipmaps`).
- **Upsampling** of pyramid levels uses `isu.upsample`. It interpolates at the true sample positions (pixel `2^L · j` for level L). `F.interpolate` would shift level L by `(2^L − 1)/2` px.
- **Blur per Level** (`kernel_map`): the Gaussian sigma that matches "pyramid level L, upsampled bicubically", computed from the pipeline's point spread function (`isu.pyramid_level_to_sigma`). Level 0 means no blur.
- **Eccentricity → Level**: piecewise linear, 0 / 2 / 3 / 4 at 4° / 8° / 16° / 24° (`ViewConfig.map_eccentricity_to_pyr_level`).

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
images/                      example scenes (sources and licenses: images/README.md)
web/                         real-time WebGPU version, see web/README.md
    index.html, main.js      viewer page and host code (JavaScript)
    shaders/                 compute and render shaders in WGSL (WebGPU Shading Language)
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{kergassner-2026,
  author = {Kerga\ss{}ner, Sophie and Didyk, Piotr},
  title = {It’s Not Just a Phase: Creating Phase-Aligned Peripheral Metamers},
  year = {2026},
  publisher = {ACM},
  doi = {10.1145/3799902.3811056},
  booktitle = {Proc. Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  articleno = {143},
  numpages = {11},
  series = {SIGGRAPH Conference Papers '26}
}
```

## License

The code is released under the [MIT License](LICENSE). The images (`images/`, `teaser.jpg`) are excluded: they are third-party works under their own licenses, listed in [images/README.md](images/README.md).
