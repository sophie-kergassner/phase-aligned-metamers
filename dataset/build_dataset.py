"""Build the aggregated calibration-study dataset from the raw response logs.

The raw logs (one .txt per participant session) are NOT part of this
repository: participants consented only to the release of aggregated results.
This script reads them from a local folder and writes only aggregated tables,
pooled over all observers. No participant IDs, dates or times are written.

Usage:
    python dataset/build_dataset.py --raw <path/to/CalibrationStudy> [--out dataset]

Outputs (in --out):
    per_image.csv      detection rate per stimulus image and mask eccentricity
    per_condition.csv  detection rate per method x level x eccentricity, pooled over scenes
    thresholds.csv     75%-correct thresholds per method x eccentricity (psychometric fit)

Cleaning (no outlier removal):
    - The first 4 trials of every session are practice trials and are dropped.
    - Trials with the 'foveated_C' stimuli of the main sessions are dropped; they
      were replaced by the extension sessions (foveated_noCE, foveated_CE).
    - Level 0 trials are dropped (not a meaningful condition, see README).
    - Each observer judged every image once per eccentricity, at a random side
      (left or right). An early bug in the experiment software occasionally
      presented an image again; only the first presentation of each
      (observer, image, eccentricity, side) is kept.

Confidence intervals:
    - Wilson score interval on the pooled trials (binomial, treats trials as independent).
    - Observer bootstrap: resample observers with replacement and recompute the
      statistic, i.e. "how would the result vary with a different group of
      observers?". Percentile interval.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm

RAW_COLUMNS = ['date', 'time', 'observer', 'scene', 'file', 'ecc', 'side', 'order', 'sequence', 'correct']
N_PRACTICE = 4
MAX_CPD = 40.0          # cycles per degree of the full-resolution image straight ahead
THRESHOLD_P = 0.75      # detection threshold of the 2AFC task (chance = 0.5)
# The released images are renamed (flat, lower case except 'CE' and 'L'): <scene>_<method>_L<level>.png
# and <scene>_reference.png (the reference was 'foveated_C_L0.png' in the study).
REFERENCE_FILE = 'reference.png'

# Stimuli shown in the main sessions as 'foveated_C_L*' had an issue and were
# replaced by the extension sessions (foveated_noCE / foveated_CE); their trials
# are excluded. foveated_C_L0 is still the reference image of every trial.
EXCLUDED_STEMS = {'foveated'}

# scene folder in the raw logs -> scene name in the dataset
SCENES = {
    'Classroom': 'classroom',
    'DrawnHouse': 'house',
    'Face': 'face',
    'Fish': 'fish',
    'ItalianFlat': 'flat',
    'MossyTree': 'tree',
}

# file stem -> method label
METHODS = {
    'foveated-noCE': 'foveated_noCE',
    'foveated-CE1gauss': 'foveated_CE',
    'orientedN_phaseN': 'intensity_adjusted',
    'orientedY_phaseN': 'oriented',
    'orientedY_phaseY': 'phase_aligned',
}


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    frames = []
    for path in sorted(raw_dir.rglob('*.txt')):
        df = pd.read_csv(path, names=RAW_COLUMNS, header=None, skipinitialspace=True)
        df['session'] = 'extension' if '_fov' in path.stem else 'main'
        df['session_file'] = path.name
        df['trial_in_session'] = np.arange(len(df))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    stats = {'sessions': df.session_file.nunique(), 'raw_trials': len(df)}

    df = df[df.trial_in_session >= N_PRACTICE]
    stats['practice_dropped'] = stats['raw_trials'] - len(df)

    key = ['observer', 'scene', 'file', 'ecc', 'side']
    n_before = len(df)
    df = df.sort_values(['session_file', 'trial_in_session']).drop_duplicates(key, keep='first')
    stats['repeats_dropped'] = n_before - len(df)

    m = df.file.str.extract(r'^(?P<stem>.+)_C_L(?P<level>\d)\.png$')
    if m.isna().any().any():
        raise ValueError(f'unexpected file names: {df.file[m.stem.isna()].unique()}')
    keep = ~m.stem.isin(EXCLUDED_STEMS)
    stats['excluded_foveated_C'] = int((~keep).sum())
    # Level 0 (full resolution) should be identical to the reference, but the
    # level-0 stimuli were generated with a contrast enhancement (foveated-CE1gauss)
    # or shown only to the first observer (foveated-noCE): no meaningful condition.
    level0 = m.level == '0'
    stats['excluded_level0'] = int((keep & level0).sum())
    keep &= ~level0
    df, m = df[keep], m[keep]
    unknown = set(m.stem) - set(METHODS)
    if unknown:
        raise ValueError(f'unknown methods: {unknown}')

    out = pd.DataFrame({
        'observer': df.observer.astype(int).values,
        'scene': df.scene.map(SCENES).values,
        'method': m.stem.map(METHODS).values,
        'level': m.level.astype(int).values,
        'mask_ecc_deg': df.ecc.astype(int).values,
        'correct': df.correct.astype(str).str.strip().eq('True').values,
    })
    out['image_file'] = out.scene + '_' + out.method + '_L' + out.level.astype(str) + '.png'
    stats['trials'] = len(out)
    stats['observers'] = out.observer.nunique()
    return out, stats


# --------------------------------------------------------------------------- statistics

def wilson(k, n, conf=0.95):
    k, n = np.asarray(k, float), np.asarray(n, float)
    z = norm.ppf(0.5 + conf / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def observer_weights(n_obs, n_boot, rng):
    """(n_boot, n_obs) multiplicities of each observer in each bootstrap sample."""
    return rng.multinomial(n_obs, np.full(n_obs, 1 / n_obs), size=n_boot).astype(float)


def aggregate(trials, group_cols, weights, observers, conf=0.95):
    """Pooled proportion per group + Wilson CI + observer-bootstrap CI."""
    obs_index = {o: i for i, o in enumerate(observers)}
    g = trials.groupby(group_cols + ['observer']).correct.agg(['sum', 'size']).reset_index()
    cells = g[group_cols].drop_duplicates().reset_index(drop=True)
    cells['cell'] = np.arange(len(cells))
    g = g.merge(cells, on=group_cols)

    K = np.zeros((len(observers), len(cells)))
    N = np.zeros_like(K)
    oi = g.observer.map(obs_index).values
    K[oi, g.cell.values] = g['sum'].values
    N[oi, g.cell.values] = g['size'].values

    k, n = K.sum(0), N.sum(0)
    lo, hi = wilson(k, n, conf)
    bk, bn = weights @ K, weights @ N
    with np.errstate(invalid='ignore', divide='ignore'):
        bp = bk / bn
    a = (1 - conf) / 2
    cells['n_trials'] = n.astype(int)
    cells['n_correct'] = k.astype(int)
    cells['n_observers'] = (N > 0).sum(0)
    cells['p_correct'] = k / n
    cells['ci95_low_wilson'] = lo
    cells['ci95_high_wilson'] = hi
    cells['ci95_low_boot'] = np.nanquantile(bp, a, axis=0)
    cells['ci95_high_boot'] = np.nanquantile(bp, 1 - a, axis=0)
    return cells.drop(columns='cell'), (K, N, cells)


# --------------------------------------------------------------------------- psychometric fit
# Same model as the paper's analysis: Weibull on log2 distance from the full
# resolution (log2(40 cpd)), from 0.5 (chance) to 1, with an anchor at 0.5 for
# the unmodified image. Least squares on per-level proportions weighted by the
# trial count equals least squares on the individual binary trials.

def weibull_inverted(x_cpd, l, k):
    x_log = np.log2(np.clip(x_cpd, 1e-10, None))
    return 1 - 0.5 * np.exp(-(((np.log2(MAX_CPD) - x_log) / l) ** k))


def fit_threshold(cpd, k, n):
    """Return the cpd at which the fitted curve reaches THRESHOLD_P (NaN on failure)."""
    keep = n > 0
    x = np.concatenate([[1e-10], cpd[keep]])
    y = np.concatenate([[0.5], k[keep] / n[keep]])
    w = np.concatenate([[1.0], n[keep]])
    try:
        (l, kk), _ = curve_fit(weibull_inverted, x, y, p0=[np.median(cpd), 3], sigma=1 / np.sqrt(w),
                               bounds=([0.1, 3], [10, 7]), maxfev=10000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan
    q = -np.log(2 * (1 - THRESHOLD_P))      # = ln 2 for 75 %
    return 2 ** (np.log2(MAX_CPD) - l * q ** (1 / kk)), l, kk


def thresholds(K, N, cells, weights, conf=0.95):
    rows = []
    a = (1 - conf) / 2
    for (method, ecc), idx in cells.groupby(['method', 'mask_ecc_deg']).groups.items():
        sub = cells.loc[idx]
        cpd = sub.cutoff_cpd.values
        Ks, Ns = K[:, sub.index], N[:, sub.index]
        t, l, kk = fit_threshold(cpd, Ks.sum(0), Ns.sum(0))
        boot = np.array([fit_threshold(cpd, w @ Ks, w @ Ns)[0] for w in weights])
        ok = np.isfinite(boot)
        rows.append({
            'method': method, 'mask_ecc_deg': ecc,
            'threshold_cpd': t,
            'ci95_low_boot_cpd': np.quantile(boot[ok], a) if ok.any() else np.nan,
            'ci95_high_boot_cpd': np.quantile(boot[ok], 1 - a) if ok.any() else np.nan,
            'boot_fit_success': ok.mean(),
            'weibull_l': l, 'weibull_k': kk,
            'n_trials': int(Ns.sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, required=True, help='folder with the raw session logs (searched recursively)')
    ap.add_argument('--out', type=Path, default=Path(__file__).parent)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    trials, stats = load_raw(args.raw)
    for k, v in stats.items():
        print(f'{k:>18}: {v}')

    rng = np.random.default_rng(args.seed)
    observers = sorted(trials.observer.unique())
    weights = observer_weights(len(observers), args.n_boot, rng)

    def add_cpd(df):
        df.insert(df.columns.get_loc('level') + 1, 'cutoff_cpd', MAX_CPD / 2 ** df.level)
        return df

    per_image, _ = aggregate(trials, ['scene', 'method', 'level', 'mask_ecc_deg', 'image_file'], weights, observers)
    per_image = add_cpd(per_image)
    per_image.insert(per_image.columns.get_loc('image_file') + 1, 'reference_file', per_image.scene + '_' + REFERENCE_FILE)
    per_image = per_image.sort_values(['scene', 'method', 'mask_ecc_deg', 'level'])

    per_cond, (K, N, cells) = aggregate(trials, ['method', 'level', 'mask_ecc_deg'], weights, observers)
    per_cond = add_cpd(per_cond).sort_values(['method', 'mask_ecc_deg', 'level'])
    cells['cutoff_cpd'] = MAX_CPD / 2 ** cells.level

    thr = thresholds(K, N, cells, weights).sort_values(['method', 'mask_ecc_deg'])

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in [('per_image', per_image), ('per_condition', per_cond), ('thresholds', thr)]:
        path = args.out / f'{name}.csv'
        df.to_csv(path, index=False, float_format='%.4f')
        print(f'wrote {path} ({len(df)} rows)')


if __name__ == '__main__':
    main()
