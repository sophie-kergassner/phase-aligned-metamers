"""Write web previews of the study stimuli: the part of each image that was on screen.

Each half of the display showed a band of 1024 image rows (per-scene offset, see
geometry.json) and the image columns 128..3968. This script crops that band from
every stimulus listed in per_image.csv and from the reference, downscales it and
saves it as WebP, for the study viewer on the project page.

Usage:
    python dataset/make_previews.py --images <path/to/released stimuli (Zenodo archive)> --out <folder> [--width 1920]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).parent


def band_box(scene_offset, size=(4096, 2048), columns=(128, 3968)):
    """Pixel box (left, top, right, bottom) of the band shown on screen; v = 0 is the bottom of the image."""
    w, h = size
    v_low, v_high = 0.5 - scene_offset, 1.0 - scene_offset
    return columns[0], round((1 - v_high) * h), columns[1], round((1 - v_low) * h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--quality', type=int, default=82)
    args = ap.parse_args()

    geo = json.load(open(HERE / 'geometry.json', encoding='utf-8'))
    offsets = geo['stimulus_layout']['scene_band_offset_o']
    per_image = pd.read_csv(HERE / 'per_image.csv')
    scene_of = {**dict(zip(per_image.image_file, per_image.scene)), **dict(zip(per_image.reference_file, per_image.scene))}
    files = sorted(scene_of)

    total = 0
    for rel in files:
        scene = scene_of[rel]
        img = Image.open(args.images / rel).convert('RGB')
        crop = img.crop(band_box(offsets[scene], img.size))
        crop = crop.resize((args.width, round(args.width * crop.height / crop.width)), Image.LANCZOS)
        dst = args.out / Path(rel).with_suffix('.webp')
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst, 'WEBP', quality=args.quality, method=6)
        total += dst.stat().st_size
    print(f'{len(files)} previews, {total / 1e6:.1f} MB, in {args.out}')


if __name__ == '__main__':
    main()
