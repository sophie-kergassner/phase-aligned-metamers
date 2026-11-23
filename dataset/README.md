# Study Data

Detection rates from the perceptual study of

> Sophie Kergaßner and Piotr Didyk. 2026. **It's Not Just a Phase: Creating Phase-Aligned Peripheral Metamers.** SIGGRAPH Conference Papers '26. <https://doi.org/10.1145/3799902.3811056>

For every stimulus image and every eccentricity: how often observers could tell the enhanced image apart from the full-resolution reference, with confidence intervals, plus the 75 % thresholds per method and the exact viewing geometry. The stimulus images themselves (full resolution PNG) are on [Zenodo](https://doi.org/10.5281/zenodo.24483038): `images.zip`, together with a copy of these tables.

| File | Contents |
|---|---|
| [`per_image.csv`](per_image.csv) | proportion correct per image (scene × method × level) and mask eccentricity, 636 rows |
| [`per_condition.csv`](per_condition.csv) | the same pooled over the six scenes (method × level × eccentricity), 107 rows |
| [`thresholds.csv`](thresholds.csv) | 75 % thresholds per method and eccentricity, 20 rows |
| [`geometry.json`](geometry.json) | display, viewing distance, fixation, stimulus layout, foveal masks, timing |
| [`plots.ipynb`](plots.ipynb) | plots of all of the above |

## Study

The setup, stimuli and methods are described in Section 6 of the paper; the exact viewing geometry is in [`geometry.json`](geometry.json). In short:

- **Task**: 2AFC, reference and test image shown in two mirrored halves; observers chose the unmodified full-resolution image. Chance is 0.5. A grey bar covered the boundary between the halves (opaque within ±40 px of the midline, fading out until ±69 px; see `separator_bar` in [`geometry.json`](geometry.json)).
- **Eccentricity** (`mask_ecc_deg`): radius of the foveal mask around the fixation point (8°, 16°, 24°); `0` is free viewing without mask.
- **Level** (`level`, `cutoff_cpd`): the input was Gaussian pyramid level *L* of the rendering, upsampled, so it contains frequencies up to `cutoff_cpd` = 40 / 2^L cpd; the methods synthesize the missing higher frequencies.
- **Design**: 11 observers; each judged every image once per eccentricity, on a random side of the display. This gives 11 trials per image and 66 per condition (6 scenes × 11 observers); see [Incomplete Cells](#incomplete-cells).

| `method` | Paper | Stimulus file |
|---|---|---|
| `foveated_noCE` | Foveated | `<scene>_foveated_noCE_L*.png` |
| `foveated_CE` | Contrast Enhanced | `<scene>_foveated_CE_L*.png` |
| `intensity_adjusted` | Intensity Adjusted | `<scene>_intensity_adjusted_L*.png` |
| `oriented` | Oriented | `<scene>_oriented_L*.png` |
| `phase_aligned` | Phase Aligned | `<scene>_phase_aligned_L*.png` |

## Columns

**`per_image.csv`**: one row per stimulus image and eccentricity.

| Column | Meaning |
|---|---|
| `scene` | `classroom`, `face`, `fish`, `flat`, `house`, `tree` |
| `method` | see table above |
| `level`, `cutoff_cpd` | pyramid level of the input and its highest frequency in cpd |
| `mask_ecc_deg` | foveal mask radius (0 = free viewing) |
| `image_file`, `reference_file` | stimulus and reference, file names in `images.zip` on Zenodo |
| `n_trials`, `n_correct`, `n_observers` | trials, correct answers, distinct observers |
| `p_correct` | `n_correct / n_trials` |
| `ci95_low_wilson`, `ci95_high_wilson` | 95 % Wilson score interval (binomial) |
| `ci95_low_boot`, `ci95_high_boot` | 95 % observer-bootstrap interval |

**`per_condition.csv`**: the same columns without `scene`, `image_file` and `reference_file`, pooled over scenes.

**`thresholds.csv`**

| Column | Meaning |
|---|---|
| `method`, `mask_ecc_deg` | condition |
| `threshold_cpd` | cutoff at which the fitted function reaches 75 % correct. Lower = the method tolerates a lower-resolution input |
| `ci95_low_boot_cpd`, `ci95_high_boot_cpd` | 95 % observer-bootstrap interval (curve refitted per sample) |
| `boot_fit_success` | fraction of bootstrap samples where the fit converged |
| `weibull_l`, `weibull_k` | fitted parameters, see below |
| `n_trials` | trials in the fit |

## Processing

[`build_dataset.py`](build_dataset.py) reads the 154 raw session logs (9484 trials) and:

1. drops the first 4 trials of every session (practice trials, 616 trials). The practice images are presented again at the end of the session, and that presentation is kept;
2. drops repeated presentations: an early version of the experiment software occasionally presented an image again; only the first presentation of each observer × image × eccentricity × side is kept (55 trials);
3. drops the trials with an earlier version of the Foveated stimuli (`foveated_C_L*`, 1605 trials), which was replaced by `foveated_noCE` and `foveated_CE`;
4. drops all level-0 trials (288 trials), i.e. full-resolution test images;
5. aggregates the remaining **6920 trials**. No outliers (observers or trials) are removed.

### Incomplete Cells

Most images have 11 trials. The number of trials of every image is plotted in [`plots.ipynb`](plots.ipynb) (section 1, *Number of Observations*); section 4 (*Every Image*) marks untested images in pink. The exceptions:

- **Last image of a stimulus block.** The generator of the trial sequences dropped the last entry of each block of the stimulus list (an off-by-one error). The base block (levels up to 5) lists the scenes in the order `tree`, `classroom`, `house`, `face`, `fish`, `flat`, so its last entry is always a `flat` image; the blocks of extra levels list them in the order `flat`, `fish`, `face`, `tree`, `classroom`, `house` and end with `house`. These images were therefore never or rarely shown: `flat` level 5 of Phase Aligned (not at 0° and 8°, 8–9 trials at 16° and 24°) and of Foveated noCE (not at 0°, 8° and 16°), `house` level 6 of Foveated noCE at 24° (not shown), `house` level 6 of Phase Aligned at 16° and level 7 at 24° (1 trial each).
- **Extra levels at 16° and 24°**: 9–10 trials, because the extra levels were added after the first two observers.
- **10 instead of 11** in a few other cells: a judgement lost to the early software bug in step 2.

**Confidence intervals.** Two kinds are given. *Wilson* treats all trials as independent. The *observer bootstrap* resamples the 11 observers with replacement (2000 samples, fixed seed) and recomputes the statistic: it answers how much the result would vary with a different group of observers. With about one trial per observer and image both are similar; when all observers answered the same way the bootstrap interval collapses to a point, while Wilson still reflects the small sample.

**Psychometric function.** As in the paper, thresholds come from a Weibull function on the log2 distance to full resolution, rising from chance to 1:

$$
p(c) = 1 - 0.5 \cdot \exp\left(-\left(\frac{\log_2 40 - \log_2 c}{l}\right)^{k}\right), \qquad 0.1 \le l \le 10, \quad 3 \le k \le 7
$$

with $c$ = `cutoff_cpd`, fitted by least squares to all trials of a condition plus an anchor point at full resolution ($p = 0.5$). The threshold is the cutoff where $p = 0.75$.

### Differences from the paper

The paper's analysis used the same trials plus 309 practice trials and 39 repeated presentations (7556 trials). The paper's fits also included the level-0 trials of Foveated CE and Foveated noCE. With the cleaning above, all thresholds change by less than 3 %, except Phase Aligned at 8° (5.57 → 5.19 cpd, −7 %): with the cleaned data, Phase Aligned tolerates an even lower-resolution input before observers detect it. All changes are well inside the confidence intervals.

## Privacy

Observers consented to the publication of aggregated results only. This folder therefore contains only numbers pooled over all observers: no observer IDs, no per-trial data, no dates or demographics. The raw logs are not public; `build_dataset.py` is included to document exactly how the tables were computed.

## License and Attribution

The data in this folder (CSV, JSON) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); please cite the paper (see [Citation](#citation)). The code (`build_dataset.py`, `plots.ipynb`) is under the repository's MIT license.

The stimuli are derived from the following scenes, whose licenses apply to the images:

| Scene | Source | License |
|---|---|---|
| `classroom` | *Classroom* by Christophe Seux, [Blender demo files](https://www.blender.org/download/demo-files/) | CC0 |
| `house` | *House of Chores* by Blender Studio, Blender demo files | CC BY |
| `fish` | *Gold* by Blender Studio, Blender demo files | CC BY |
| `flat` | *Italian Flat* by Flavio Della Tommasa, Blender demo files | CC BY |
| `tree` | *Tree, leaves and grass* by Simon Thommes, Blender Studio, Blender demo files | CC0 |
| `face` | [photo on Unsplash](https://unsplash.com/photos/woman-holding-her-face-hHCJcg4ZxWQ) by Wei Ding | [Unsplash License](https://unsplash.com/license) |

## Citation

If you use this data, please cite our paper:

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
