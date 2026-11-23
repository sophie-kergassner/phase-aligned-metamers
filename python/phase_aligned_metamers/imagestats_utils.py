import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

import torchvision

def import_image_NCHW(path, max_value=None):
    """
    Loads an image as a [1, C, H, W] float tensor in [0, 1].

    max_value: value that is mapped to 1. Default: from the bit depth (255 for 8-bit, 65535 for 16-bit images).
               The calibration study stimuli were generated with 16-bit renderings divided by 255 * 255 = 65025
               (about 0.8% too bright); pass max_value=255*255 to reproduce them exactly.
    """
    image = torchvision.io.decode_image(path)
    if max_value is None:
        max_value = 65535 if image.dtype == torch.uint16 else 255
    image = image.to(torch.int32).float() / max_value  # via int32: not all ops support uint16
    return image.unsqueeze(0)

def rgb_2_ycrcb(image):
        """This function is a direct copy of odak.learn.perception.color_conversion"""
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        ycrcb = torch.zeros(image.size()).to(image.device)
        ycrcb[:, 0, :, :] =       0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        ycrcb[:, 1, :, :] = 0.5 + 0.713 * (image[:, 0, :, :] - ycrcb[:, 0, :, :])
        ycrcb[:, 2, :, :] = 0.5 + 0.564 * (image[:, 2, :, :] - ycrcb[:, 0, :, :])
        return ycrcb

def ycrcb_2_rgb(image):
    """This function is a direct copy of odak.learn.perception.color_conversion"""
    if len(image.shape) == 3:
       image = image.unsqueeze(0)
    if image.shape[1] == 1:  # grayscale: Y is the gray value
        return image.repeat(1, 3, 1, 1)
    rgb = torch.zeros(image.size(), device=image.device)
    rgb[:, 0, :, :] = image[:, 0, :, :] + 1.403 * (image[:, 1, :, :] - 0.5)
    rgb[:, 1, :, :] = image[:, 0, :, :] - 0.714 * (image[:, 1, :, :] - 0.5) - 0.344 * (image[:, 2, :, :] - 0.5)
    rgb[:, 2, :, :] = image[:, 0, :, :] + 1.773 * (image[:, 2, :, :] - 0.5)
    return rgb

# default display width [px] of a full image in the notebook; rows and plots use the same width.
# 680 px matches a code line of 88 characters in VS Code's default font (14 px Consolas, ~7.7 px per character).
DISPLAY_WIDTH = 680


NAN_GRAY = 0.4  # display color of NaN pixels (neutral gray, distinct from the colormaps' ends)


def _prepare_display(img, color='inferno', vmin=None, vmax=None, vsymmetric=False):
    """
    Converts a tensor ([N,C,H,W], [C,H,W] or [H,W]; C = 1 or 3) into an 8-bit RGB array for display.
    Single-channel images are mapped with the colormap `color` between vmin and vmax (default: their min/max);
    RGB images are clipped to [0, 1]. NaN pixels of single-channel images are shown in NAN_GRAY and ignored for
    the value range, e.g. to hide regions where a statistic is not defined. Returns (rgb_uint8, info dict).
    """
    import matplotlib
    t = img.detach().float().cpu()
    while t.dim() > 3:
        t = t[0]
    if t.dim() == 3 and t.shape[0] == 1:
        t = t[0]
    finite = torch.isfinite(t)
    data_min = t[finite].min().item() if finite.any() else 0.0
    data_max = t[finite].max().item() if finite.any() else 0.0

    if t.dim() == 3:  # RGB
        rgb = t[:3].permute(1, 2, 0).clamp(0, 1).numpy()
        info = dict(h=rgb.shape[0], w=rgb.shape[1], min=data_min, max=data_max, scalar=False)
        return (rgb * 255).round().astype(np.uint8), info

    if vsymmetric:
        abs_max = max(abs(data_min), abs(data_max))
        vmin, vmax = -abs_max, abs_max
    lo = data_min if vmin is None else vmin
    hi = data_max if vmax is None else vmax
    norm = ((t.numpy() - lo) / (hi - lo)) if hi > lo else np.zeros(t.shape)
    cmap = matplotlib.colormaps[color]
    rgb = cmap(np.clip(np.nan_to_num(norm), 0, 1))[..., :3]
    rgb[~finite.numpy()] = NAN_GRAY
    info = dict(h=t.shape[0], w=t.shape[1], min=data_min, max=data_max, scalar=True, vmin=lo, vmax=hi, cmap=cmap)
    return (rgb * 255).round().astype(np.uint8), info


def _encode_image(rgb, max_px=1200, quality=92):
    """8-bit RGB array -> (data URI, image-rendering CSS value). Large images as JPEG, small ones as PNG."""
    import base64, io
    from PIL import Image as PILImage
    pil = PILImage.fromarray(rgb)
    if max(pil.size) > max_px:  # keep the notebook small; the browser scales the image anyway
        scale = max_px / max(pil.size)
        pil = pil.resize((round(pil.size[0] * scale), round(pil.size[1] * scale)), PILImage.LANCZOS)
    # small images are upscaled by the browser with smoothing; 'pixelated' keeps their pixels visible
    rendering = 'pixelated' if max(pil.size) < 256 else 'auto'
    buf = io.BytesIO()
    if max(pil.size) <= 512:   # small images (e.g. coarse pyramid levels): lossless
        pil.save(buf, format='PNG')
        mime = 'png'
    else:                      # large images: JPEG keeps the notebook small (PNG is ~10x larger here)
        pil.save(buf, format='JPEG', quality=quality)
        mime = 'jpeg'
    return f'data:image/{mime};base64,' + base64.b64encode(buf.getvalue()).decode(), rendering


# Caption title: one line, truncated with an ellipsis if it does not fit (full text in the hover tooltip).
# Used as a flex item (min-width:0 lets it shrink); the size label next to it gets flex-shrink:0.
TITLE_CSS = 'font-weight:600; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap'


def _figure_html(rgb, info, title, colorbar=True, max_px=1200):
    """HTML block (plain divs; GitHub strips <figure>/<figcaption>): the image (scaled to the available width) and a text caption below it."""
    import html
    src, rendering = _encode_image(rgb, max_px)

    fmt = lambda v: f'{v:.3g}'
    mono = "font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:11px; opacity:0.65; white-space:nowrap"
    title_txt = html.escape(str(title)) if title else ""
    head = (f'<div style="display:flex; justify-content:space-between; align-items:baseline; gap:8px">'
            f'<span title="{title_txt}" style="{TITLE_CSS}">{title_txt}</span>'
            f'<span style="{mono}; flex-shrink:0">{info["h"]} &times; {info["w"]}</span></div>')
    if info['scalar'] and colorbar:
        stops = ', '.join('rgb({},{},{})'.format(*[int(255 * c) for c in info['cmap'](x)[:3]]) for x in np.linspace(0, 1, 9))
        clipped = info['min'] < info['vmin'] - 1e-9 or info['max'] > info['vmax'] + 1e-9
        data_note = f'<span style="{mono}">data [{fmt(info["min"])}, {fmt(info["max"])}]</span>' if clipped else ''
        legend = (f'<div style="display:flex; align-items:center; gap:6px; margin-top:4px">'
                  f'<span style="{mono}">{fmt(info["vmin"])}</span>'
                  f'<div style="flex:1; height:6px; border-radius:3px; background:linear-gradient(to right, {stops})"></div>'
                  f'<span style="{mono}">{fmt(info["vmax"])}</span>{data_note}</div>')
    else:  # RGB: no colormap, so only the value range
        legend = f'<div style="margin-top:2px"><span style="{mono}">range [{fmt(info["min"])}, {fmt(info["max"])}]</span></div>'
    return (f'<div style="margin:0; min-width:0">'
            f'<img src="{src}" style="width:100%; height:auto; display:block; border-radius:2px; image-rendering:{rendering}">'
            f'<div style="font-family:sans-serif; font-size:13px; line-height:1.35; margin-top:5px">'
            f'{head}{legend}</div></div>')


def _wrap(html_str, suptitle=''):
    """Optional heading above the output."""
    if not suptitle:
        return html_str
    head = (f'<div style="margin-bottom:8px; font-family:sans-serif; font-size:14px; font-weight:600">'
            f'<span>{suptitle}</span></div>')
    return head + html_str


# Placeholder colour for matplotlib figures shown with show_figure(): it is replaced by the text colour of the
# notebook theme (CSS currentColor), so axes and labels are dark in light themes and light in dark themes.
THEME_COLOR = '#010203'


def show_figure(fig, title='', note='', max_width=None):
    """
    Displays a matplotlib figure as inline SVG (sharp at any size, transparent background) with the same caption
    style as show(): `title` in bold, `note` (e.g. an equation) as small monospace text on the right.
    Everything drawn in isu.THEME_COLOR takes the text colour of the notebook theme. Closes the figure.
    Outside Jupyter, the figure is shown with plt.show().
    """
    import html, io, re, uuid
    max_width = DISPLAY_WIDTH if max_width is None else max_width
    buf = io.StringIO()
    with plt.rc_context({'svg.fonttype': 'path', 'svg.hashsalt': uuid.uuid4().hex}):  # unique ids per figure
        fig.savefig(buf, format='svg', transparent=True, bbox_inches='tight', pad_inches=0.08)
    svg = buf.getvalue()
    svg = svg[svg.index('<svg'):]                                    # drop XML header / doctype
    svg = svg.replace(THEME_COLOR, 'currentColor')
    svg = re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"', r'<svg\1 width="100%" height="auto"', svg, count=1)
    mono = "font-family:ui-monospace,Consolas,monospace; font-size:11px; opacity:0.65; white-space:nowrap"
    out = (f'<div style="margin:0; max-width:{int(max_width)}px">'
           f'<div style="line-height:0">{svg}</div>'
           f'<div style="font-family:sans-serif; font-size:13px; line-height:1.35; margin-top:5px; '
           f'display:flex; justify-content:space-between; align-items:baseline; gap:8px">'
           f'<span title="{html.escape(title)}" style="{TITLE_CSS}">{html.escape(title)}</span>'
           f'<span style="{mono}; flex-shrink:0">{html.escape(note)}</span>'
           f'</div></div>')
    if _display_html(out):
        plt.close(fig)
    else:
        plt.show()


def _display_html(html_str):
    """Displays HTML in Jupyter; returns False outside of IPython."""
    try:
        from IPython import get_ipython
        from IPython.display import display, HTML
        if get_ipython() is None:
            return False
        display(HTML(html_str))
        return True
    except ImportError:
        return False


def _show_matplotlib(arrays, titles, max_columns=None):
    """Fallback outside Jupyter: plain matplotlib figure without axes."""
    n = len(arrays)
    cols = n if not max_columns else min(max_columns, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, a, t in zip(axes.flat, arrays, titles):
        ax.imshow(a)
        ax.set_title(t or '', fontsize=8)
    plt.tight_layout()
    plt.show()


def show(img:torch.Tensor, color='inferno', vmin=None, vmax=None, title=None, max_width=None, showColorbar=True, vsymmetric=False):
    """
    Displays one image: RGB ([..,3,H,W]) as is (clipped to [0, 1]), single-channel images with the colormap `color`.
    In Jupyter, the image is shown without axes and scaled to the available width (up to max_width px; default
    DISPLAY_WIDTH for RGB images, 420 for colormapped maps); title,
    size and value range are shown as text below it, plus a colour scale for single-channel images (showColorbar).
    To save an image, use save_png().
    """
    rgb, info = _prepare_display(img, color, vmin, vmax, vsymmetric)
    figure = _figure_html(rgb, info, title, colorbar=showColorbar)
    if max_width is None:
        max_width = 420 if info['scalar'] else DISPLAY_WIDTH
    if not _display_html(f'<div style="max-width:{int(max_width)}px">{figure}</div>'):
        _show_matplotlib([rgb], [title])


def show_multiple(pyr, color='inferno', title:list=None, suptitle="", vmin=None, vmax=None, max_columns=None, min_width=None, max_width=None, row_width=None, vsymmetric=False, showColorbar=True):
    """
    Displays several images side by side (see show()). `color` and `title` can be lists (one entry per image).
    The grid wraps on narrow screens (each image at least min_width px; default: 300 if the row contains an RGB image,
    220 for colormapped maps only); max_columns fixes the number of columns instead. Each image is at most
    max_width px wide (default: 420 with an RGB image in the row, 340 otherwise).
    row_width: fixed total width of the row in px (e.g. isu.DISPLAY_WIDTH); all images in one row share it equally
    (or max_columns per row). Overrides min_width / max_width.
    suptitle: optional heading above the row.
    vsymmetric: color range symmetric around 0 (per image), e.g. for differences or noise; a list sets it per image.
    """
    if len(pyr) == 0:
        return
    arrays, figures, titles, any_rgb = [], [], [], False
    for i, img in enumerate(pyr):
        c = (color[i] if i < len(color) else 'inferno') if isinstance(color, list) else color
        t = (title[i] if i < len(title) else '') if title is not None else ''
        vs = (vsymmetric[i] if i < len(vsymmetric) else False) if isinstance(vsymmetric, list) else vsymmetric
        rgb, info = _prepare_display(img, c, vmin, vmax, vs)
        arrays.append(rgb)
        titles.append(t)
        any_rgb |= not info['scalar']
        figures.append(_figure_html(rgb, info, t, colorbar=showColorbar))

    n = len(pyr)
    if max_width is None:
        max_width = 420 if any_rgb else 340
    if min_width is None:
        min_width = 300 if any_rgb else 220
    n_cols = min(max_columns, n) if max_columns else n
    if row_width is not None:
        cols = f'repeat({n_cols}, minmax(0, 1fr))'
        max_width = int(row_width)
    else:
        if max_columns:
            cols = f'repeat({n_cols}, minmax(0, 1fr))'
        else:
            cols = f'repeat(auto-fit, minmax({int(min_width)}px, 1fr))'
        max_width = int(n_cols * max_width + (n_cols - 1) * 14)  # total grid width (14 px gaps)
    grid = f'<div style="display:grid; grid-template-columns:{cols}; gap:14px; max-width:{max_width}px">{"".join(figures)}</div>'
    if not _display_html(_wrap(grid, suptitle)):
        _show_matplotlib(arrays, titles, max_columns)


def show_compare(images, titles, title='', color='gray', vmin=None, vmax=None, max_width=None):
    """
    Interactive comparison (Jupyter / VS Code): drag the slider to wipe between image A (left of the line) and
    image B (right), or use the buttons to show A or B completely. With more than two images, A and B can be
    chosen from drop-down menus (default: A = first, B = last image).

    images: list of tensors of the same size; titles: one title per image
    Single-channel images use the colormap `color` with a shared range (vmin/vmax; default: joint min/max).
    """
    import html, json, uuid
    max_width = DISPLAY_WIDTH if max_width is None else max_width
    if vmin is None:
        vmin = min(im.min().item() for im in images)
    if vmax is None:
        vmax = max(im.max().item() for im in images)
    prepared = [_prepare_display(im, color, vmin, vmax) for im in images]
    srcs = [_encode_image(rgb)[0] for rgb, _ in prepared]
    info = prepared[0][1]
    rendering = 'pixelated' if max(info['h'], info['w']) < 256 else 'auto'
    a, b = 0, len(images) - 1
    uid = 'cmp' + uuid.uuid4().hex[:8]

    wipe = (f"var c=document.getElementById('{uid}'),v=VAL;"
            f"c.querySelector('img.a').style.clipPath='inset(0 '+(100-v)+'% 0 0)';"
            f"c.querySelector('.line').style.left=v+'%';c.querySelector('input').value=v;")
    pick = (f"var c=document.getElementById('{uid}');var s=JSON.parse(c.dataset.srcs);"
            f"c.querySelector('img.WHICH').src=s[this.value];")
    btn = ('font-family:sans-serif; font-size:12px; padding:2px 10px; border:1px solid rgba(128,128,128,0.6); '
           'border-radius:4px; background:transparent; color:inherit; cursor:pointer')
    sel = ('font-family:sans-serif; font-size:12px; padding:1px 4px; border:1px solid rgba(128,128,128,0.6); '
           'border-radius:4px; background:transparent; color:inherit; max-width:48%')
    mono = "font-family:ui-monospace,Consolas,monospace; font-size:11px; opacity:0.65; flex-shrink:0"

    def chooser(which, default):
        if len(images) == 2:
            return f'<span title="{html.escape(titles[default])}" style="{TITLE_CSS}">{html.escape(titles[default])}</span>'
        opts = ''.join(f'<option value="{i}"{" selected" if i == default else ""}>{html.escape(t)}</option>' for i, t in enumerate(titles))
        return f'<select style="{sel}" onchange="{pick.replace("WHICH", which)}">{opts}</select>'

    widget = (f'<div id="{uid}" data-srcs="{html.escape(json.dumps(srcs))}" style="font-family:sans-serif; font-size:13px; min-width:0">'
              f'<div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:4px">'
              f'{chooser("a", a)}{chooser("b", b)}</div>'
              f'<div style="position:relative; line-height:0">'
              f'<img class="b" src="{srcs[b]}" style="width:100%; display:block; image-rendering:{rendering}">'
              f'<img class="a" src="{srcs[a]}" style="position:absolute; top:0; left:0; width:100%; image-rendering:{rendering}; clip-path:inset(0 50% 0 0)">'
              f'<div class="line" style="position:absolute; top:0; bottom:0; left:50%; width:2px; margin-left:-1px; '
              f'background:rgba(255,255,255,0.85); box-shadow:0 0 2px rgba(0,0,0,0.6)"></div></div>'
              f'<input type="range" min="0" max="100" value="50" style="width:100%; margin:6px 0 2px" oninput="{wipe.replace("VAL", "this.value")}">'
              f'<div style="display:flex; justify-content:space-between; align-items:center; gap:8px">'
              f'<button style="{btn}" onclick="{wipe.replace("VAL", "100")}">show A</button>'
              f'<span style="{mono}">{info["h"]} &times; {info["w"]}</span>'
              f'<button style="{btn}" onclick="{wipe.replace("VAL", "0")}">show B</button></div></div>')

    out = f'<div style="max-width:{int(max_width)}px">{widget}</div>'
    if not _display_html(_wrap(out, title)):
        _show_matplotlib([rgb for rgb, _ in prepared], titles)


def nh_generate_preblurred_image(img: np.ndarray, max_value: int):
    """
    Generate a pre-blurred image for all levels
    :param img:
    :param max_value:
    :return:
    """
    levels = np.empty((max_value + 1, *img.shape[1:]), dtype=np.float32)  # Pre-allocate memory
    levels[0] = img.detach().cpu().numpy()
    # Apply Gaussian filters for all levels
    for i in range(max_value):
        sigma = np.sqrt((i+1)**2 - i**2)  # Calculate the equivalent sigma for the 2D Gaussian
        gaussian_filter(levels[i], sigma=(sigma), output=levels[i+1], truncate=2)  # Store directly in `levels`

    return levels

def nh_foveate_image(img, sigma_map):
    """Per-pixel Gaussian blur: blends the two pre-blurred images whose (integer) sigmas enclose sigma_map."""
    max_level = sigma_map.max().int() + 1
    levels = nh_generate_preblurred_image(img, max_level)
    levels = torch.tensor(levels, device=img.device)

    sigma_map_int = sigma_map.int()
    sigma_residue = sigma_map - sigma_map_int

    interpolated = torch.gather(levels, 0, sigma_map_int.to(dtype=torch.int64)) * (1 - sigma_residue) + torch.gather(levels, 0, (sigma_map_int + 1).to(dtype=torch.int64)) * sigma_residue

    return interpolated

def get_gaussian_kernel(sigma, kernel_size=None, device='cpu'):
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)

    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


# Blur kernel used for every step of the Gaussian pyramid (in the pixels of the level that is blurred).
PYR_KERNEL_SIGMA = 1.0
PYR_KERNEL_SIZE = 5


def effective_kernel_sigma(sigma=PYR_KERNEL_SIGMA, kernel_size=PYR_KERNEL_SIZE):
    """
    Standard deviation of the discrete, truncated and normalized Gaussian kernel from get_gaussian_kernel.
    Truncation removes the tails, so it is slightly smaller than `sigma` (5x5, sigma=1 -> 0.961).
    """
    x = np.arange(kernel_size) - kernel_size // 2
    w = np.exp(-x**2 / (2 * sigma**2))
    w /= w.sum()
    return float(np.sqrt((w * x**2).sum()))


def pyramid_sigma_formula(level, sigma0=None):
    """
    Sigma [full-resolution px] of the Gaussian blur that is equivalent to the *downsampling* part of the pyramid,
    i.e. going down `level` steps (without the blur added by upsampling the level again).

    Step k blurs with sigma0 in the pixels of level k, i.e. with sigma0 * 2**k in full-resolution pixels.
    Gaussian blurs add up in variance, so after L steps:

        sigma_L^2 = sigma0^2 * sum_{k=0}^{L-1} 4^k = sigma0^2 * (4^L - 1) / 3

    -> sigma_0 = 0 (level 0 is the unblurred image), sigma_1 = sigma0, sigma_2 = sqrt(5)*sigma0, ...

    level:  number or tensor
    sigma0: per-step sigma; default: effective sigma of the pyramid kernel (PYR_KERNEL_SIGMA, PYR_KERNEL_SIZE)
    """
    if sigma0 is None:
        sigma0 = effective_kernel_sigma()
    return sigma0 * ((4.0 ** level - 1) / 3) ** 0.5


_PYR_SIGMA_CACHE = {}

def pyramid_sigma_measured(max_level, upsampling='bicubic', sigma=PYR_KERNEL_SIGMA, kernel_size=PYR_KERNEL_SIZE):
    """
    Equivalent Gaussian sigma [full-resolution px] of 'pyramid level L, upsampled back to full resolution'
    (get_gauss_mipmaps + F.interpolate(mode=upsampling)), for the integer levels 0..max_level.

    Measured, not fitted to any image: pyramid + upsampling is a linear operation (without gamma), so we pass a
    single impulse through it at each of the 2**L positions of the level-L sampling grid and average the
    responses -> mean point spread function (PSF). Its frequency response (MTF) drops to 50% at f50; the
    Gaussian with the same f50 (MTF(f) = exp(-2 pi^2 sigma^2 f^2)) has

        sigma = sqrt(ln(2) / 2) / (pi * f50)

    MTF50 is used instead of the PSF variance because bicubic interpolation has negative lobes, which make the
    variance underestimate the visible blur. The MTF ignores spatial shifts. Results are cached.
    """
    key = (max_level, upsampling, sigma, kernel_size)
    if key in _PYR_SIGMA_CACHE:
        return _PYR_SIGMA_CACHE[key]

    N = 1024 * 2**max(0, max_level - 4)       # 1D signal length, long enough for the widest PSF
    H = 4 * 2**max_level                        # height, so that every level is >= 4 px high (reflect padding)
    f = np.fft.rfftfreq(N)
    sigmas = [0.0]
    for L in range(1, max_level + 1):
        psf = np.zeros(N)
        for p in range(2**L):
            x = torch.zeros(1, 1, H, N)
            x[..., N // 2 + p] = 1.0             # vertical line = 1D impulse
            level = get_gauss_mipmaps(x, L + 1, sigma=sigma, kernel_size=kernel_size)[L]
            up = F.interpolate(level, size=(H, N), mode=upsampling)  # as in the study; the shift vs upsample() does not change the MTF
            psf += np.roll(up[0, 0, H // 2].numpy(), -p)   # align all responses to the impulse position
        psf /= psf.sum()
        mtf = np.abs(np.fft.rfft(np.fft.ifftshift(psf)))
        i = int(np.argmax(mtf < 0.5))
        f50 = f[i-1] + (0.5 - mtf[i-1]) * (f[i] - f[i-1]) / (mtf[i] - mtf[i-1])
        sigmas.append(float(np.sqrt(np.log(2) / 2) / (np.pi * f50)))

    _PYR_SIGMA_CACHE[key] = np.array(sigmas)
    return _PYR_SIGMA_CACHE[key]


def pyramid_level_to_sigma(level):
    """
    Sigma [full-resolution px] per (possibly fractional) pyramid level, e.g. for a per-pixel level map.

    Matches 'pyramid level L, upsampled with F.interpolate(mode='bicubic')', i.e. how the stimuli of the
    calibration study were generated (see pyramid_sigma_measured). For fractional levels, the formula is scaled
    by the measured/formula ratio, interpolated between the neighbouring integer levels, so sigma stays smooth
    and is 0 at level 0.
    """
    formula = pyramid_sigma_formula(level)
    is_tensor = torch.is_tensor(level)
    lv = level.detach().float().cpu().numpy() if is_tensor else np.asarray(level, dtype=np.float64)
    max_level = max(1, int(np.ceil(lv.max())))
    measured = pyramid_sigma_measured(max_level)
    levels = np.arange(max_level + 1)
    ratio = np.ones(max_level + 1)
    ratio[1:] = measured[1:] / pyramid_sigma_formula(levels[1:])
    ratio[0] = ratio[1]                          # formula is 0 at level 0 anyway
    scale = np.interp(lv, levels, ratio)

    if is_tensor:
        return formula * torch.as_tensor(scale, dtype=formula.dtype, device=level.device)
    return formula * scale


def upsample(x, size, mode='bicubic'):
    """
    Upsamples a level of the Gaussian pyramid (get_gauss_mipmaps) to `size` at the correct sample positions.

    The pyramid keeps every 2nd pixel, so sample j of a level that is 2**L times smaller lies at pixel 2**L * j
    of the target. F.interpolate(align_corners=False) instead assumes 2**L * j + (2**L - 1) / 2, which shifts the
    result by (2**L - 1) / 2 px (0.5, 1.5, 3.5, ... px for L = 1, 2, 3); align_corners=True stretches it instead.
    Here, every target pixel x is sampled at level position x / 2**L (bicubic/bilinear/nearest via grid_sample;
    pixels beyond the last sample are clamped to the border).

    x:    [N, C, h, w] real or complex tensor
    size: target (H, W); the factor 2**L per axis is derived from the size ratio
    """
    if torch.is_complex(x):
        return torch.complex(upsample(x.real, size, mode), upsample(x.imag, size, mode))

    h, w = x.shape[-2:]
    H, W = int(size[0]), int(size[1])
    if (h, w) == (H, W):
        return x
    fy = 2.0 ** round(np.log2(H / h))
    fx = 2.0 ** round(np.log2(W / w))

    # sample positions in the level's pixel units, normalized to [-1, 1] for grid_sample(align_corners=True)
    gy = torch.arange(H, device=x.device, dtype=x.dtype) / fy
    gx = torch.arange(W, device=x.device, dtype=x.dtype) / fx
    gy = 2 * gy / max(h - 1, 1) - 1
    gx = 2 * gx / max(w - 1, 1) - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing='ij')[::-1], dim=-1)  # (H, W, 2) as (x, y)
    grid = grid.unsqueeze(0).expand(x.shape[0], H, W, 2)
    return F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)


def get_gauss_mipmaps(img, levels, sigma=PYR_KERNEL_SIGMA, kernel_size=PYR_KERNEL_SIZE):
    """
    Builds a Gaussian pyramid with `levels` levels. Level 0 is `img` itself; each further
    level is blurred with a Gaussian and downsampled by 2. The values are blurred as they are (no gamma),
    as for the calibration study stimuli.

    Downsampling keeps every 2nd pixel ([::2]), so sample j of level L lies at full-resolution pixel 2**L * j.
    Use upsample() (not F.interpolate) to bring levels back to full resolution at the correct positions.

    img: [N, C, H, W] tensor
    """
    C = img.shape[1]
    kernel = get_gaussian_kernel(sigma=sigma, kernel_size=kernel_size, device=img.device)
    kernel = kernel.to(img.dtype).repeat(C, 1, 1, 1)  # one kernel per channel: [C, 1, k, k]
    pad = kernel_size // 2

    mipmap = [img]
    for _ in range(1, levels):
        level = F.pad(mipmap[-1], (pad, pad, pad, pad), mode='reflect')
        level = F.conv2d(level, kernel, groups=C)
        mipmap.append(level[:, :, ::2, ::2])

    return mipmap

def compose_rgb(fov_ycbcr, noise=None, gain=1.0):
    """
    Adds `noise` (times `gain`) to the luma channel of the foveated image and converts the result to RGB in [0, 1].
    Works for color ([N,3,H,W] YCbCr) and grayscale ([N,1,H,W]) images.
    """
    ycc = fov_ycbcr.clone()
    if noise is not None:
        ycc[:, 0:1, :, :] += gain * noise
    return ycrcb_2_rgb(ycc).clamp(min=0, max=1)


def save_png(img, path, alpha=None, crop=None, color=None, vmin=None, vmax=None):
    """
    Saves an image tensor with values in [0, 1] (clamped) as an 8-bit PNG/JPG; the format follows the file extension.

    img:   [N,C,H,W], [C,H,W] or [H,W]; C = 1 (gray) or 3 (RGB). Only the first image of a batch is saved.
    alpha: optional mask of the same spatial size ([N,1,H,W], [1,H,W] or [H,W]) -> saved as RGBA (PNG only).
    crop:  optional (y0, y1, x0, x1) region to save.
    color: optional matplotlib colormap for single-channel images (e.g. 'twilight' for angles), applied between
           vmin and vmax (default: the image's min/max). Without it, single-channel images are saved as gray.
    Parent folders are created if needed.

    Examples:
        save_png(final_img_rgb, 'out/final.png')
        save_png(noise_enhancement + 0.5, 'out/noise.png')                       # signed noise around mid-gray
        save_png(maps.y[0]['phase_rep'] / (2 * torch.pi) + 0.5, 'out/phase.png')  # angle in [-pi, pi] -> [0, 1]
        save_png(torch.ones_like(mask), 'out/mask.png', alpha=mask)              # white image, mask as transparency
    """
    import os
    from PIL import Image as PILImage

    def to_chw(t):
        t = t.detach().float().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t

    img = to_chw(img)
    if img.shape[0] == 1 and color is not None:
        img = torch.from_numpy(_prepare_display(img, color, vmin, vmax)[0]).permute(2, 0, 1).float() / 255
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if alpha is not None:
        img = torch.cat([img, to_chw(alpha)[0:1]], dim=0)
    if crop is not None:
        y0, y1, x0, x1 = crop
        img = img[:, y0:y1, x0:x1]

    array = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    PILImage.fromarray(array).save(path)  # RGB or RGBA, inferred from the channel count
