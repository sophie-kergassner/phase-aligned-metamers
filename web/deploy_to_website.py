"""
Copies the WebGPU demo into the website's project folder, so that the page can load it from there:

    <website>/projects/phase-aligned-metamers/realtime/
        main.js, constants.json, shaders/*.wgsl, images/*.jpg

The page itself (phase-aligned-metamers.html) contains the demo's markup and loads realtime/main.js as a module;
main.js finds its shaders and constants via data-base on the element with [data-pam-demo].

Usage (from the repository root):
    python web/deploy_to_website.py <path to the website repository> [--no-images]
"""
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..')
SCENES = ['classroom', 'drawnhouse', 'face', 'fish', 'italianflat', 'mossytree']


def main(website, copy_images=True):
    target = os.path.join(website, 'projects', 'phase-aligned-metamers', 'realtime')
    os.makedirs(os.path.join(target, 'shaders'), exist_ok=True)
    copied = []
    for f in ['main.js', 'constants.json']:
        shutil.copy2(os.path.join(HERE, f), os.path.join(target, f))
        copied.append(f)
    for f in glob.glob(os.path.join(HERE, 'shaders', '*.wgsl')):
        shutil.copy2(f, os.path.join(target, 'shaders', os.path.basename(f)))
        copied.append('shaders/' + os.path.basename(f))
    if copy_images:
        os.makedirs(os.path.join(target, 'images'), exist_ok=True)
        for scene in SCENES:
            src = os.path.join(REPO, 'images', f'{scene}.jpg')
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target, 'images', os.path.basename(src)))
                copied.append('images/' + os.path.basename(src))
            else:
                print(f'missing: {src}')
    print(f'copied {len(copied)} files to {target}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        sys.exit(__doc__)
    main(args[0], copy_images='--no-images' not in sys.argv)
