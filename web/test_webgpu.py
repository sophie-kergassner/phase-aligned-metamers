"""
Runs the WebGPU shaders of the web version (web/shaders/*.wgsl) headless with wgpu-py and compares every stage
with the Python package, which is the reference implementation.

Requirements: pip install wgpu   (plus the package's own requirements)
Usage (from the repository root):
    python web/test_webgpu.py [path/to/image] [--uniform=LEVEL] [--analyse-foveated] [--write-constants]

--uniform=LEVEL      one pyramid level for the whole image (e.g. 2.5), which exercises a smaller cutoff
--analyse-foveated   pyramid and analysis of the foveated image (Worm(img.fov)) instead of the original
--fit=cutoff         intensity fit as in the paper (default: weighted, see Worm.compute_intensity_modulators)
--measured           noise from the measured statistics (no repair; the web version's "Noise From: Measured")
--keep=I,D,P         effect of the parameters: keep_intensity, keep_dir, keep_phase (NoiseGenerator.randomize_stats),
                     with the web version's random fields (hash, see web_rand) instead of torch.rand
--no-align           without the phase convention alignment (Worm(align_phase_convention=False), the paper's
                     behaviour; GPU: RepairParams.align = 0); default: aligned on both sides
--write-constants    rewrite web/constants.json from the package

The host logic here mirrors web/main.js (same textures, bind group layouts, dispatch order), so a passing test
means the shaders compute the same as the package; main.js only has to drive them identically.
"""
import json
import os
import struct
import sys

import numpy as np
import torch
import wgpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import phase_aligned_metamers.imagestats_utils as isu
import phase_aligned_metamers.phase_analysis_utils as phu
from phase_aligned_metamers.image import Image
from phase_aligned_metamers.noise_generator import NoiseGenerator
from phase_aligned_metamers.view_config import ViewConfig
from phase_aligned_metamers.worm import Worm

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = '--no-align' not in sys.argv   # phase convention alignment (fixes/1_phase_sign_alignment.md), on both sides
SCREEN_WIDTH_CM, SCREEN_WIDTH_PX, VIEWING_DISTANCE = 120, 3840, 100
MAX_PYR_DEPTH = 5
ANALYSIS_LEVELS = MAX_PYR_DEPTH + 1   # levels 0..5: cutoff (<= 4) + 1 extra level for the phase repair
NOISE_LEVELS = MAX_PYR_DEPTH - 1      # noise bands for levels 0..3
MIN_LEVEL_SIZE = 8                    # Worm.MIN_LEVEL_SIZE
FIT_LEVELS = 3                        # Worm(fit_levels=3)

U = wgpu.TextureUsage
B = wgpu.BufferUsage
COMPUTE, FRAGMENT = wgpu.ShaderStage.COMPUTE, wgpu.ShaderStage.FRAGMENT


# ----------------------------------------------------------------------------------------------- constants
def constants_from_package():
    """Everything the browser needs from the package (written to web/constants.json)."""
    sigma0 = isu.effective_kernel_sigma()
    measured = isu.pyramid_sigma_measured(7)
    formula = np.array([isu.pyramid_sigma_formula(L) for L in range(8)])
    ratios = np.ones(8)
    ratios[1:] = measured[1:] / formula[1:]
    ratios[0] = ratios[1]
    pyr_kernel = isu.get_gaussian_kernel(isu.PYR_KERNEL_SIGMA, isu.PYR_KERNEL_SIZE)
    filters = torch.cat([f.reshape(-1) for f in phu.get_filters()])
    return {
        'note': 'From the Python package; written by web/test_webgpu.py --write-constants. '
                'sigma0 = isu.effective_kernel_sigma(); ratios = isu.pyramid_sigma_measured / isu.pyramid_sigma_formula '
                'per integer level 0..7 (ratio[0] = ratio[1]); pyr_kernel = isu.get_gaussian_kernel (5x5, row-major); '
                'filters = phu.get_filters() [Ga, Gb, Gc, Ha, Hb, Hc, Hd], 9x9 each, row-major.',
        'sigma0': float(sigma0),
        'ratios': [float(r) for r in ratios],
        'pyr_kernel': [float(v) for v in pyr_kernel.reshape(-1)],
        'filters': [float(v) for v in filters],
    }


def check_constants_file(current):
    path = os.path.join(HERE, 'constants.json')
    if '--write-constants' in sys.argv or not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(current, f, indent=1)
        print(f'wrote {path}')
    stored = json.load(open(path, encoding='utf-8'))
    ok = all(np.allclose(np.atleast_1d(stored[k]), np.atleast_1d(current[k]), atol=0, rtol=0)
             for k in ['sigma0', 'ratios', 'pyr_kernel', 'filters'])
    print(f'  {"constants.json":36s} {"matches the package" if ok else "OUTDATED: run with --write-constants"}')
    return ok


# ----------------------------------------------------------------------------------------------- GPU helpers
class Gpu:
    def __init__(self):
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
        self.device = self.adapter.request_device_sync()

    def shader(self, *names):
        code = '\n'.join(open(os.path.join(HERE, 'shaders', n + '.wgsl'), encoding='utf-8').read() for n in names)
        return self.device.create_shader_module(code=code)

    def pipeline(self, names, entries, entry='main'):
        bgl = self.device.create_bind_group_layout(entries=entries)
        pipe = self.device.create_compute_pipeline(layout=self.device.create_pipeline_layout(bind_group_layouts=[bgl]),
                                                   compute={'module': self.shader(*names), 'entry_point': entry})
        return pipe, bgl

    def bind_group(self, bgl, resources):
        entries = []
        for i, r in resources:
            if isinstance(r, wgpu.GPUBuffer):
                r = {'buffer': r, 'offset': 0, 'size': r.size}
            elif isinstance(r, wgpu.GPUTexture):
                r = r.create_view()
            entries.append({'binding': i, 'resource': r})
        return self.device.create_bind_group(layout=bgl, entries=entries)

    def run(self, pipe_bgl, resources, w, h, groups=None):
        """One dispatch (8 x 8 workgroups); resources: list for bindings 0..n-1, or list of (binding, resource)."""
        pipe, bgl = pipe_bgl
        if resources and not isinstance(resources[0], tuple):
            resources = list(enumerate(resources))
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, self.bind_group(bgl, resources))
        cp.dispatch_workgroups(*(groups or ((w + 7) // 8, (h + 7) // 8)), 1)
        cp.end()
        self.device.queue.submit([enc.finish()])

    def texture(self, w, h, fmt, usage=U.TEXTURE_BINDING | U.STORAGE_BINDING | U.COPY_SRC, layers=1):
        return self.device.create_texture(size=(w, h, layers), format=fmt, usage=usage)

    def buffer(self, data, usage):
        return self.device.create_buffer_with_data(data=data, usage=usage)

    def uniform(self, *vals):
        return self.buffer(np.array(vals, dtype=np.float32).tobytes(), B.UNIFORM)

    def read(self, tex, channels=4):
        w, h = tex.size[0], tex.size[1]
        bpr = (w * 4 * channels + 255) // 256 * 256
        data = self.device.queue.read_texture({'texture': tex, 'origin': (0, 0, 0)},
                                              {'offset': 0, 'bytes_per_row': bpr, 'rows_per_image': h}, (w, h, 1))
        return np.frombuffer(data, dtype=np.float32).reshape(h, bpr // 4)[:, :w * channels].reshape(h, w, channels)

    def read_buffer(self, buf):
        return np.frombuffer(self.device.queue.read_buffer(buf), dtype=np.float32)


def tex_entry(binding, sample_type='unfilterable-float', dim='2d', vis=COMPUTE):
    return {'binding': binding, 'visibility': vis, 'texture': {'sample_type': sample_type, 'view_dimension': dim}}


def tex_entries(first, last, vis=COMPUTE):
    return [tex_entry(i, vis=vis) for i in range(first, last + 1)]


def storage_entry(binding, fmt):
    return {'binding': binding, 'visibility': COMPUTE,
            'storage_texture': {'access': 'write-only', 'format': fmt, 'view_dimension': '2d'}}


def uniform_entry(binding, vis=COMPUTE):
    return {'binding': binding, 'visibility': vis, 'buffer': {'type': 'uniform'}}


def ro_entry(binding, vis=COMPUTE):
    return {'binding': binding, 'visibility': vis, 'buffer': {'type': 'read-only-storage'}}


def rw_entry(binding):
    return {'binding': binding, 'visibility': COMPUTE, 'buffer': {'type': 'storage'}}


def params_array(w, h, consts, gaze, viewer, max_layer, view_mode=1, uniform_level=-1.0, aux_level=0, aux_scale=1.0, cutoff=0):
    r = consts['ratios']
    return np.array([w, h, SCREEN_WIDTH_PX / SCREEN_WIDTH_CM, consts['sigma0'],
                     gaze[0], gaze[1], uniform_level, max_layer,
                     viewer[0], viewer[1], viewer[2], view_mode,
                     *r[:4], *r[4:8],
                     w, h, 0, 1,   # disp.w: noise amp. 1
                     aux_level, aux_scale, 0, cutoff], dtype=np.float32)


# ----------------------------------------------------------------------------------------------- reports
def report(name, gpu, ref, tol):
    d = np.abs(np.asarray(gpu, dtype=np.float64) - np.asarray(ref, dtype=np.float64))
    if d.size == 0:
        print(f'  {name:36s} skipped (no pixels)')
        return True
    ok = bool(d.max() <= tol)
    print(f'  {name:36s} max |diff| {d.max():.2e}   mean {d.mean():.2e}   (tolerance {tol:g})  {"OK" if ok else "MISMATCH"}')
    return ok


def report_angle(name, gpu, ref, period, weight, tol, share=0.9999):
    """Angles modulo their period, where they are well defined (weight above 1% of its max); passes if `share` of
    those pixels are within tol (single pixels with a nearly undefined angle may differ arbitrarily)."""
    d = np.abs((gpu - ref + period / 2) % period - period / 2)[weight > 0.01 * weight.max()]
    q = np.quantile(d, share)
    ok = bool(q <= tol)
    print(f'  {name:36s} {100 * share:.2f}% within {q:.2e}   max {d.max():.2e}   (tolerance {tol:g}, '
          f'{100 * d.size / weight.size:.0f}% of px well defined)  {"OK" if ok else "MISMATCH"}')
    return ok


def report_share(name, gpu, ref, tol, share=0.999, period=None):
    """Passes if `share` of all pixels are within tol; prints the maximum and the number of outliers too."""
    d = np.asarray(gpu, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    d = np.abs((d + period / 2) % period - period / 2) if period else np.abs(d)
    q = np.quantile(d, share)
    ok = bool(q <= tol)
    print(f'  {name:36s} {100 * share:.1f}% of px within {q:.2e}   max {d.max():.2e}   (tolerance {tol:g})  {"OK" if ok else "MISMATCH"}')
    print(f'      px off by > 10 x tolerance: {int((d > 10 * tol).sum())} ({100 * (d > 10 * tol).mean():.4f}%)')
    return ok


# ----------------------------------------------------------------------------------------------- reference
def weighted_fit(worm, level_map):
    """Reference for the web version's weighted intensity fit (web/shaders/lap_stats, lap_finalize, fit): line through
    the std of ALL Laplacian bands, each measured only where its level is fully kept (mask 1 - clamp(level - i, 0, 1)
    == 1, nearest-sampled at the band's size) and weighted by that fraction. Replaces the Worm's (paper) modulators."""
    import torch.nn.functional as F
    xs, ys, ws = [], [], []
    for i in range(len(worm.y) - 1):
        band = worm.y[i]['laplace']
        q = F.interpolate((1 - (level_map - i).clamp(0, 1) >= 1).float(), size=band.shape[2:], mode='nearest')
        if q.sum() < 1:
            continue
        mean = (q * band).sum() / q.sum()
        xs.append(i); ys.append(torch.sqrt((q * (band - mean) ** 2).sum() / q.sum()).item()); ws.append(q.mean().item())
    x, y, wt = np.array(xs, dtype=np.float64), np.array(ys), np.array(ws)
    A = np.stack((x, np.ones_like(x)), axis=1)
    m, b = np.linalg.solve(A.T @ (wt[:, None] * A), A.T @ (wt * y))
    for i, layer in enumerate(worm.y[:worm.depth + worm.analysis_pyr_overhead]):
        layer['intensity_modulator'] = max(m * i + b, 0.0)
        layer['intensity'] = layer['G-H-complex'].abs() * layer['intensity_modulator']
    worm.fit = {'levels': x, 'std': y, 'weights': wt, 'm': m, 'b': b}


def pcg(v):
    """PCG hash on uint32 arrays, as pcg() in web/shaders/noise_band.wgsl (wraps modulo 2^32)."""
    v = v.astype(np.uint32)
    state = v * np.uint32(747796405) + np.uint32(2891336453)
    word = ((state >> ((state >> np.uint32(28)) + np.uint32(4))) ^ state) * np.uint32(277803737)
    return (word >> np.uint32(22)) ^ word


def web_rand(h, w, s, k):
    """Random field of the web version for level s (size h x w) and statistic k (0 intensity, 1 direction, 2 phase)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.uint32)
    hsh = pcg(pcg(pcg(x) + y) + np.uint32(s * 4 + k))
    return torch.from_numpy((hsh >> np.uint32(8)).astype(np.float64) / 16777216.0).float()[None, None]


def randomize_web(worm, keep):
    """NoiseGenerator.randomize_stats with the web version's random fields, applied to the repaired statistics.
    (The GPU's phase tracking starts from the current orientation, so in a single frame it changes nothing.)"""
    ki, kd, kp = keep
    def cyclical_blend(a, b, k):
        return (k * torch.polar(torch.ones_like(a), a) + (1 - k) * torch.polar(torch.ones_like(b), b)).angle()
    for s in range(worm.cutoff):
        lv = worm.y[s]
        h, w = lv['intensity_rep'].shape[2:]
        u = [web_rand(h, w, s, k) for k in range(3)]
        intensity, d, ph = lv['intensity_rep'], lv['dir_rep'], lv['phase_rep']
        lv['intensity_rep'] = ki * intensity + (1 - ki) * u[0] * intensity.std()
        lv['dir_rep'] = cyclical_blend(d * 2, (u[1] * torch.pi - torch.pi) * 2, kd) / 2
        lv['phase_rep'] = cyclical_blend(ph, u[2] * 2 * torch.pi - torch.pi, kp)


def reference(img_path, vc, analysis_src=None, intensity_fit='weighted', keep=None, measured=False):
    """The Python pipeline as in example.ipynb. analysis_src: image to analyse instead of the original (e.g. the
    foveated one); the test passes the GPU's own foveated image, so that both analyse exactly the same input."""
    img = Image('cpu', img_path, vc.kernel_map)
    h, w = img.ycbcr.shape[2:]
    worm = Worm('cpu', img.ycbcr if analysis_src is None else analysis_src, vc.pyr_level_map, MAX_PYR_DEPTH, align_phase_convention=ALIGN)
    worm_orig = worm if analysis_src is None else Worm('cpu', img.ycbcr, vc.pyr_level_map, MAX_PYR_DEPTH, align_phase_convention=ALIGN)
    worm_orig.compute_foveated_imgs()
    for step in ['compute_R', 'compute_theta', 'compute_steered_G_H', 'compute_phase']:
        worm.y = getattr(worm, step)(worm.y)
    analysis = {i: {k: worm.y[i][k].clone() for k in ['dir', 'phase', 'S']} | {'mag': worm.y[i]['G-H-complex'].abs()}
                for i in range(worm.depth + worm.analysis_pyr_overhead)}
    worm.y = worm.compute_intensity_modulators(worm.y)   # the paper's fit
    if intensity_fit == 'weighted':                      # the web version's default fit (only there, not in the package)
        weighted_fit(worm, vc.pyr_level_map)
    for step in ['add_repaired_layer_proxies'] + ([] if measured else ['repair_intensity', 'repair_orientation', 'repair_phase']):
        worm.y = getattr(worm, step)(worm.y)
    # repaired statistics before the blend (the GPU blends inside noise_band.wgsl, its img.rep stays unblended)
    worm.rep_unblended = [{k: worm.y[i][k].clone() for k in ['intensity_rep', 'dir_rep', 'phase_rep']} for i in range(worm.cutoff)]
    if keep is not None:
        randomize_web(worm, keep)
    noise_gen = NoiseGenerator([1, 1, h, w], device='cpu')
    worm.y = noise_gen.generate_noise(worm.y, worm.cutoff)
    for i in range(worm.cutoff):   # as compose.wgsl: an all-zero band (std 0) adds nothing (the package would give NaN)
        if not torch.isfinite(noise_gen.noises[i]).all():
            noise_gen.noises[i] = torch.zeros_like(noise_gen.noises[i])
    noise = noise_gen.get_noise_enhancement(worm.a, worm.cutoff)
    return dict(img=img, worm=worm, worm_orig=worm_orig, analysis=analysis,
                noise_gen=noise_gen, noise=noise, final=isu.compose_rgb(img.fov, noise))


# ----------------------------------------------------------------------------------------------- main
def main(img_path, uniform_level=None, analyse_foveated=False, intensity_fit='weighted', keep=None, measured=False):
    consts = constants_from_package()
    rgb_u8 = isu.torchvision.io.decode_image(img_path)[:3]          # same decoder as the package
    h, w = rgb_u8.shape[1:]
    vc = ViewConfig('cpu', SCREEN_WIDTH_CM, SCREEN_WIDTH_PX, [1, 1, h, w], VIEWING_DISTANCE, uniform_level=uniform_level)
    ul = -1.0 if uniform_level is None else float(uniform_level)

    gpu = Gpu()
    dev = gpu.device
    max_layer = int(float(isu.pyramid_level_to_sigma(torch.tensor(4.0)))) + 1
    n_levels = int(np.floor(np.log2(min(h, w) / MIN_LEVEL_SIZE))) + 1
    n_analysis = min(ANALYSIS_LEVELS, n_levels)
    print(f'GPU: {gpu.adapter.info["device"]} ({gpu.adapter.info["backend_type"]}) | image {w} x {h} | '
          f'analysis of the {"foveated" if analyse_foveated else "original"} image | {intensity_fit} fit' +
          (f' | uniform level {uniform_level}' if uniform_level is not None else '') +
          (f' | keep intensity / dir / phase {keep}' if keep is not None else ''))

    # ---------------- pipelines (as main.js)
    P = {
        'ycbcr': gpu.pipeline(['ycbcr'], [tex_entry(0, 'float'), storage_entry(1, 'rgba32float')]),
        'blur_v': gpu.pipeline(['blur'], [uniform_entry(0), tex_entry(1), storage_entry(2, 'rgba32float')], 'blur_v'),
        'blur_h': gpu.pipeline(['blur'], [uniform_entry(0), tex_entry(1), storage_entry(2, 'rgba32float'), storage_entry(3, 'rgba16float')], 'blur_h'),
        'down': gpu.pipeline(['sampling', 'pyramid_down'], [ro_entry(0), tex_entry(1), storage_entry(2, 'rgba32float')]),
        'laplace': gpu.pipeline(['sampling', 'laplace'], [*tex_entries(0, 1), storage_entry(2, 'r32float')]),
        'filters': gpu.pipeline(['sampling', 'analysis_filters'], [ro_entry(0), tex_entry(1), *[storage_entry(i, 'rgba32float') for i in (2, 3, 4)]]),
        'orient': gpu.pipeline(['analysis_orient'], [*tex_entries(0, 2), storage_entry(3, 'rgba32float')]),
        'foveate': gpu.pipeline(['common', 'foveate'], [uniform_entry(0), tex_entry(1), tex_entry(2, dim='2d-array'),
                                                        storage_entry(3, 'rgba32float'), storage_entry(4, 'rgba32float')]),
        'foveate_pyr': gpu.pipeline(['common', 'sampling', 'foveate_pyramid'], [uniform_entry(0), *tex_entries(1, 6), storage_entry(7, 'rgba32float')]),
        'stats': gpu.pipeline(['sampling', 'noise_stats'], [uniform_entry(0), *tex_entries(1, 2), rw_entry(3)]),
        'finalize': gpu.pipeline(['noise_finalize'], [uniform_entry(0), ro_entry(1), rw_entry(2)]),
        'lap_stats': gpu.pipeline(['lap_stats'], [uniform_entry(0), *tex_entries(1, 2), rw_entry(3)]),
        'lap_finalize': gpu.pipeline(['lap_finalize'], [uniform_entry(0), ro_entry(1), rw_entry(2)]),
        'fit': gpu.pipeline(['fit'], [uniform_entry(0), ro_entry(1), rw_entry(2)]),
        'rep_init': gpu.pipeline(['sampling', 'repair_common', 'repair_init'], [uniform_entry(0), tex_entry(1), storage_entry(2, 'rgba32float'), ro_entry(15)]),
        'rep_phi': gpu.pipeline(['sampling', 'repair_common', 'repair_phi'], [uniform_entry(0), *tex_entries(1, 4), storage_entry(5, 'rgba32float'), tex_entry(6), tex_entry(7)]),
        'rep_level': gpu.pipeline(['sampling', 'repair_common', 'repair_level'], [uniform_entry(0), *tex_entries(1, 5), storage_entry(6, 'rgba32float'), tex_entry(7), ro_entry(15)]),
        'sem': gpu.pipeline(['sampling', 'repair_common', 'semantics'], [uniform_entry(0), *tex_entries(1, 4), storage_entry(5, 'rgba32float')]),
        'semH': gpu.pipeline(['sem_smooth'], [tex_entry(0), storage_entry(1, 'rgba32float')], 'sem_h'),
        'semV': gpu.pipeline(['sem_smooth'], [tex_entry(0), storage_entry(1, 'rgba32float')], 'sem_v'),
        'band': gpu.pipeline(['noise_band'], [uniform_entry(0), ro_entry(1), tex_entry(2), storage_entry(3, 'r32float'), ro_entry(4), tex_entry(5)]),
        'track': gpu.pipeline(['noise_band'], [uniform_entry(0), tex_entry(2), tex_entry(6), storage_entry(7, 'r32float')], 'track'),
        'compose': gpu.pipeline(['common', 'sampling', 'compose'], [uniform_entry(0), uniform_entry(1), *tex_entries(2, 8), ro_entry(9),
                                                                     storage_entry(10, 'rgba32float'), ro_entry(11)]),
    }
    kern = gpu.buffer(np.array(consts['pyr_kernel'], dtype=np.float32).tobytes(), B.STORAGE)
    filt = gpu.buffer(np.array(consts['filters'], dtype=np.float32).tobytes(), B.STORAGE)

    def std_of(tex, factor, size_tex, out_buf, index):
        """noise_stats + noise_finalize: std of `tex` upsampled by `factor` to the size of `size_tex`."""
        sw, sh = size_tex.size[0], size_tex.size[1]
        gx, gy = (sw + 15) // 16, (sh + 15) // 16
        partial = dev.create_buffer(size=gx * gy * 8, usage=B.STORAGE)
        gpu.run(P['stats'], [gpu.uniform(factor, gx, 0, 0), tex, size_tex, partial], 0, 0, groups=(gx, gy))
        gpu.run(P['finalize'], [gpu.uniform(gx * gy, sw * sh, index, 0), partial, out_buf], 0, 0, groups=(1, 1))

    def build_pyramid(level0):
        """Gaussian pyramid (YCbCr), Laplacian bands (Y) and analysis of a full-resolution YCbCr texture."""
        levels, sizes = [level0], [(w, h)]
        for i in range(1, n_levels):
            lw, lh = (sizes[-1][0] + 1) // 2, (sizes[-1][1] + 1) // 2
            lvl = gpu.texture(lw, lh, 'rgba32float')
            gpu.run(P['down'], [kern, levels[-1], lvl], lw, lh)
            levels.append(lvl)
            sizes.append((lw, lh))
        laplace = []
        for i in range(n_levels - 1):
            lap = gpu.texture(*sizes[i], 'r32float')
            gpu.run(P['laplace'], [levels[i], levels[i + 1], lap], *sizes[i])
            laplace.append(lap)
        analysis = []
        for i in range(n_analysis):
            ra, rb, cc, out = (gpu.texture(*sizes[i], 'rgba32float') for _ in range(4))
            gpu.run(P['filters'], [filt, levels[i], ra, rb, cc], *sizes[i])
            gpu.run(P['orient'], [ra, rb, cc, out], *sizes[i])
            analysis.append(out)
        return levels, sizes, laplace, analysis

    # ---------------- per image
    src = gpu.texture(w, h, 'rgba8unorm', U.TEXTURE_BINDING | U.COPY_DST)
    rgba = torch.cat([rgb_u8, torch.full((1, h, w), 255, dtype=torch.uint8)]).permute(1, 2, 0).contiguous().numpy()
    dev.queue.write_texture({'texture': src, 'origin': (0, 0, 0)}, rgba.tobytes(),
                            {'offset': 0, 'bytes_per_row': w * 4, 'rows_per_image': h}, (w, h, 1))
    ycc, tmp, ping, pong = (gpu.texture(w, h, 'rgba32float') for _ in range(4))
    stack = gpu.texture(w, h, 'rgba16float', U.TEXTURE_BINDING | U.STORAGE_BINDING, layers=max_layer + 1)
    gpu.run(P['ycbcr'], [src, ycc], w, h)
    cur = ycc
    for k in range(1, max_layer + 1):
        sigma = float(np.sqrt(k ** 2 - (k - 1) ** 2))
        ub = gpu.buffer(struct.pack('<fiII', sigma, int(2 * sigma + 0.5), 0, 0), B.UNIFORM)
        nxt = ping if cur is not ping else pong
        gpu.run(P['blur_v'], [ub, cur, tmp], w, h)
        gpu.run(P['blur_h'], [ub, tmp, nxt, stack.create_view(dimension='2d', base_array_layer=k, array_layer_count=1)], w, h)
        cur = nxt
    orig_levels, sizes, orig_laplace, orig_analysis = build_pyramid(ycc)

    # ---------------- per frame (gaze at the image centre, as the reference)
    maps, fov, fov_pyr, final = (gpu.texture(w, h, 'rgba32float') for _ in range(4))
    pbuf = gpu.buffer(params_array(w, h, consts, vc.gaze_pos_px, vc.viewer_pos_cm, max_layer, uniform_level=ul).tobytes(), B.UNIFORM)
    gpu.run(P['foveate'], [pbuf, ycc, stack.create_view(dimension='2d-array'), maps, fov], w, h)
    gpu.run(P['foveate_pyr'], [pbuf, *[orig_levels[min(i, n_levels - 1)] for i in range(6)], fov_pyr], w, h)
    if analyse_foveated:   # main.js: pyramid and analysis of the base image, every frame
        levels, sizes, laplace, analysis = build_pyramid(fov)
    else:
        levels, laplace, analysis = orig_levels, orig_laplace, orig_analysis

    # reference: in the foveated mode it analyses the GPU's foveated image (float16 blur stack, see below)
    g_fov_t = torch.from_numpy(gpu.read(fov)[..., :3].copy()).permute(2, 0, 1)[None]
    R = reference(img_path, vc, g_fov_t if analyse_foveated else None, intensity_fit, keep, measured)
    img, worm = R['img'], R['worm']

    # Laplacian std per band and intensity fit on the GPU
    weighted = 1.0 if intensity_fit == 'weighted' else 0.0
    lap_stats = dev.create_buffer(size=64, usage=B.STORAGE | B.COPY_SRC)
    for i, lap in enumerate(laplace):
        gx, gy = (lap.size[0] + 15) // 16, (lap.size[1] + 15) // 16
        partial = dev.create_buffer(size=gx * gy * 16, usage=B.STORAGE)
        gpu.run(P['lap_stats'], [gpu.uniform(i, gx, weighted, 0), lap, maps, partial], 0, 0, groups=(gx, gy))
        gpu.run(P['lap_finalize'], [gpu.uniform(gx * gy, i, weighted, 0), partial, lap_stats], 0, 0, groups=(1, 1))
    cut = int(np.ceil(vc.pyr_level_map.max().item() - 1e-6))   # main.js: from the image border
    fit_buf = dev.create_buffer(size=48, usage=B.STORAGE | B.COPY_SRC)
    gpu.run(P['fit'], [gpu.uniform(cut, len(laplace), weighted, 0), lap_stats, fit_buf], 0, 0, groups=(1, 1))

    # repair
    reps = [gpu.texture(*sizes[i], 'rgba32float') for i in range(n_analysis)]
    phis = [gpu.texture(*sizes[i], 'rgba32float') for i in range(n_analysis)]
    for i in range(0 if measured else cut, n_analysis):
        gpu.run(P['rep_init'], [(0, gpu.uniform(i, 0, 0, 0)), (1, analysis[i]), (2, reps[i]), (15, fit_buf)], *sizes[i])
    for i in range(-1 if measured else cut - 1, -1, -1):
        gpu.run(P['rep_phi'], [gpu.uniform(i, 0, int(ALIGN), 0), analysis[i + 1], reps[i + 1], analysis[i + 2], reps[i + 2], phis[i + 1], analysis[i + 1], maps], *sizes[i + 1])   # 6, 7: Fix 4 inputs (off: unused)
        gpu.run(P['rep_level'], [(0, gpu.uniform(i, 0, int(ALIGN), 0)), (1, maps), (2, analysis[i]), (3, reps[i + 1]), (4, analysis[i + 1]),
                                 (5, phis[i + 1]), (6, reps[i]), (7, analysis[i]), (15, fit_buf)], *sizes[i])

    # noise bands, their std at full resolution, composition
    bands = [gpu.texture(*sizes[i], 'r32float') for i in range(NOISE_LEVELS)]
    band_stds = dev.create_buffer(size=32, usage=B.STORAGE | B.COPY_SRC)
    # effect of the parameters: std of intensity_rep per level (plain std: lap_stats / lap_finalize, unweighted)
    rep_std = dev.create_buffer(size=64, usage=B.STORAGE | B.COPY_SRC)
    kp = (1.0, 1.0, 1.0) if keep is None else keep
    randomize = 1.0 if min(kp) < 1.0 else 0.0
    for i in range(cut if randomize else 0):
        gx, gy = (sizes[i][0] + 15) // 16, (sizes[i][1] + 15) // 16
        partial = dev.create_buffer(size=gx * gy * 16, usage=B.STORAGE)
        gpu.run(P['lap_stats'], [gpu.uniform(i, gx, 0, 0), reps[i], maps, partial], 0, 0, groups=(gx, gy))
        gpu.run(P['lap_finalize'], [gpu.uniform(gx * gy, i, 0, 0), partial, rep_std], 0, 0, groups=(1, 1))
    for i in range(cut):
        # first frame, as main.js: phase tracking reset (state = current orientation), which leaves the phase unchanged
        bp = gpu.uniform(h, w, i, randomize, *kp, 2)
        track_prev, track = gpu.texture(*sizes[i], 'r32float'), gpu.texture(*sizes[i], 'r32float')
        gpu.run(P['track'], [(0, bp), (2, reps[i]), (6, track_prev), (7, track)], *sizes[i])
        gpu.run(P['band'], [bp, filt, reps[i], bands[i], rep_std, track], *sizes[i])
        std_of(bands[i], 2 ** i, ycc, band_stds, i)
    gpu.run(P['compose'], [pbuf, gpu.uniform(cut, 0, 1, 0), maps, fov, fov_pyr, *bands, band_stds, final, fit_buf], w, h)

    # display: views rendered offscreen at image resolution (gaze marker off), as main.js on the canvas
    disp_mod = gpu.shader('common', 'sampling', 'display')
    d_bgl = dev.create_bind_group_layout(entries=[uniform_entry(0, FRAGMENT), *tex_entries(1, 6, FRAGMENT), ro_entry(7, FRAGMENT), ro_entry(8, FRAGMENT)])
    d_pipe = dev.create_render_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[d_bgl]),
                                        vertex={'module': disp_mod, 'entry_point': 'vs'},
                                        fragment={'module': disp_mod, 'entry_point': 'fs', 'targets': [{'format': 'rgba8unorm'}]},
                                        primitive={'topology': 'triangle-list'})

    def render_view(mode, aux=None, aux_level=0, aux_scale=1.0):
        target = gpu.texture(w, h, 'rgba8unorm', U.RENDER_ATTACHMENT | U.COPY_SRC)
        vbuf = gpu.buffer(params_array(w, h, consts, vc.gaze_pos_px, vc.viewer_pos_cm, max_layer, view_mode=mode, uniform_level=ul,
                                       aux_level=aux_level, aux_scale=aux_scale, cutoff=cut).tobytes(), B.UNIFORM)
        enc = dev.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{'view': target.create_view(), 'load_op': 'clear', 'store_op': 'store', 'clear_value': (0, 0, 0, 1)}])
        rp.set_pipeline(d_pipe)
        rp.set_bind_group(0, gpu.bind_group(d_bgl, list(enumerate([vbuf, ycc, fov, maps, fov_pyr, analysis[0] if aux is None else aux,
                                                                    final, band_stds, fit_buf]))))
        rp.draw(3)
        rp.end()
        dev.queue.submit([enc.finish()])
        return np.frombuffer(dev.queue.read_texture({'texture': target, 'origin': (0, 0, 0)},
                                                    {'offset': 0, 'bytes_per_row': w * 4, 'rows_per_image': h}, (w, h, 1)),
                             dtype=np.uint8).reshape(h, w, 4)[..., :3].astype(np.float32)

    # ---------------- compare
    as_hwc = lambda t: t[0].permute(1, 2, 0).numpy()
    tol = 1e-4   # values in about [-1, 1]; sums over many taps differ by float32 rounding (~1e-5)
    # The per-pixel foveation uses a float16 blur stack on the GPU (up to 5e-4 off from img.fov). In the foveated
    # analysis, the reference therefore analyses the GPU's foveated image; the effect of the float16 stack on the
    # result is reported separately at the end.
    fov_tol = 2e-3
    ok = check_constants_file(consts)

    print('milestone 1: geometry and per-pixel foveation')
    ok &= report('YCbCr', gpu.read(ycc)[..., :3], as_hwc(img.ycbcr), 1e-6)
    g_maps = gpu.read(maps)
    ecc_ref = vc.ecc_map[0, 0].numpy()
    near = ecc_ref < 1.0   # acos is ill-conditioned near 0 deg
    ok &= report('eccentricity [deg] (> 1 deg)', g_maps[..., 0][~near], ecc_ref[~near], 1e-3)
    ok &= report('eccentricity [deg] (< 1 deg)', g_maps[..., 0][near], ecc_ref[near], 5e-2)
    ok &= report('pyramid level', g_maps[..., 1], vc.pyr_level_map[0, 0].numpy(), 1e-3)
    ok &= report('blur sigma [px] (from same level)', g_maps[..., 2],
                 isu.pyramid_level_to_sigma(torch.from_numpy(g_maps[..., 1].copy())).numpy(), 1e-4)
    ok &= report('blur sigma [px]', g_maps[..., 2], vc.kernel_map[0, 0].numpy(), 5e-3)   # sqrt(level) near level 0
    g_fov = gpu.read(fov)[..., :3]
    ok &= report('foveated (blur)', g_fov, as_hwc(img.fov), fov_tol)
    fovea = vc.pyr_level_map[0, 0].numpy() == 0
    ok &= report('foveated (blur), fovea only', g_fov[fovea], as_hwc(img.fov)[fovea], 1e-6)
    ok &= report('displayed RGB [0..255]', render_view(1), isu.ycrcb_2_rgb(img.fov).clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255, 1.0)

    print('milestone 2: pyramid, foveation from the pyramid, analysis, intensity fit')
    wo = R['worm_orig']
    ok &= report('foveated (pyramid)', gpu.read(fov_pyr)[..., :3], as_hwc(torch.cat([wo.y[0]['fov'], wo.cb[0]['fov'], wo.cr[0]['fov']], dim=1)), tol)
    t2 = tol
    n_ref = min(n_levels, len(worm.y))   # the Worm builds only cutoff + 1 + fit levels
    for i in range(n_ref):
        ref = torch.cat([worm.y[i]['img'], worm.cb[i]['img'], worm.cr[i]['img']], dim=1)
        ok &= report(f'pyramid level {i} (YCbCr)', gpu.read(levels[i])[..., :3], as_hwc(ref), t2)
    for i in range(n_ref - 1):
        ok &= report(f'Laplacian band {i}', gpu.read(laplace[i], channels=1)[..., 0], worm.y[i]['laplace'][0, 0].numpy(), t2)
    g_stats = gpu.read_buffer(lap_stats).reshape(8, 2)
    fit_ref = worm.fit   # levels, std, weights of the points of the fit
    lv = fit_ref['levels'].astype(int)
    ok &= report(f'band std of the fit points (relative)', g_stats[lv, 0] / fit_ref['std'], 1.0, 1e-4)
    if intensity_fit == 'weighted':
        ok &= report('band weights (kept fraction)', g_stats[lv, 1], fit_ref['weights'], 1e-5)
    for i in range(min(n_analysis, len(R['analysis']))):
        a, ref = gpu.read(analysis[i]), R['analysis'][i]
        ok &= report_angle(f'L{i} orientation [rad]', a[..., 0], ref['dir'][0, 0].numpy(), np.pi, ref['S'][0, 0].numpy(), 1e-3)
        ok &= report_angle(f'L{i} phase [rad]', a[..., 1], ref['phase'][0, 0].numpy(), 2 * np.pi, ref['mag'][0, 0].numpy(), 1e-3)
        mag_ref = ref['mag'][0, 0].numpy()
        ok &= report_share(f'L{i} magnitude |G + iH| (rel. to max)', a[..., 2] / mag_ref.max(), mag_ref / mag_ref.max(), tol, share=0.9999)
    print(f'    (cutoff L{cut} as Worm: {cut == worm.cutoff})')
    ok &= (cut == worm.cutoff)
    ok &= report('noise amplitude per level (fit)', gpu.read_buffer(fit_buf)[:cut + 1],
                 [float(worm.y[i]['intensity_modulator']) for i in range(cut + 1)], 1e-5)

    print('milestone 3: repair (levels below the cutoff)')
    # The phase extrapolation doubles the angle per level (2 * angle(c_up)), so float32 rounding differences double
    # too (L3 -> L0: ~1e-5 -> ~2e-3). A few isolated pixels, where Python's upscale_complex_phase multiplies by the
    # sign of an angle close to +-pi, flip; both results are equally valid there.
    angle_tol = 5e-3
    for i in range(cut):
        r, ref = gpu.read(reps[i]), worm.rep_unblended[i]
        ok &= report_share(f'L{i} intensity_rep', r[..., 0], ref['intensity_rep'][0, 0].numpy(), t2)
        ok &= report_share(f'L{i} dir_rep [rad]', r[..., 1], ref['dir_rep'][0, 0].numpy(), angle_tol, period=np.pi)
        ok &= report_share(f'L{i} phase_rep [rad]', r[..., 2], ref['phase_rep'][0, 0].numpy(), angle_tol, period=2 * np.pi)
    if keep is not None:
        g_rep_std = gpu.read_buffer(rep_std)
        for i in range(cut):
            ok &= report(f'L{i} std of intensity_rep (relative)', g_rep_std[2 * i] / worm.rep_unblended[i]['intensity_rep'].std().item(), 1.0, 1e-4)

    print('milestone 4: noise synthesis and composition')
    g_band_stds = gpu.read_buffer(band_stds)
    ng = R['noise_gen']
    for i in range(cut):
        ref_band = ng.noises_small[i][0, 0].numpy()
        if np.abs(ref_band).max() == 0:   # all-zero band: the GPU's must be zero as well
            g_band = gpu.read(bands[i], channels=1)[..., 0]
            ok &= report(f'noise band L{i} (all zero)', np.abs(g_band).max(), 0.0, 1e-12)
            continue
        ok &= report_share(f'noise band L{i} (relative)', gpu.read(bands[i], channels=1)[..., 0] / np.abs(ref_band).max(),
                           ref_band / np.abs(ref_band).max(), 1e-3)
        std_ref = isu.upsample(ng.noises_small[i], (h, w)).std().item()
        ok &= report(f'noise band L{i} std (relative)', g_band_stds[i] / std_ref, 1.0, 1e-4)
    g_final = gpu.read(final)
    n_nan = int(np.isnan(g_final).sum())
    ok &= n_nan == 0
    print(f'  final image NaN values: {n_nan}  {"OK" if n_nan == 0 else "MISMATCH"}')
    noise_ref = R['noise'][0, 0].numpy()
    ok &= report_share('noise enhancement (rel. to max)', g_final[..., 3] / np.abs(noise_ref).max(), noise_ref / np.abs(noise_ref).max(), 1e-3)
    final_rgb = isu.ycrcb_2_rgb(torch.from_numpy(g_final[..., :3].copy()).permute(2, 0, 1)[None]).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    ok &= report_share('final RGB', final_rgb, R['final'][0].permute(1, 2, 0).numpy(), 2e-3)
    ok &= report_share('displayed final RGB [0..255]', render_view(15), R['final'][0].permute(1, 2, 0).numpy() * 255, 1.0)
    for i in range(cut):   # display mode 18: contribution of level i to the noise enhancement
        contrib = (ng.noises[i] * (1 - R['worm'].a[i]['mask_FR']))[0, 0].numpy()
        shown_i = render_view(18, aux=bands[i], aux_level=i, aux_scale=4.0)[..., 0]
        ok &= report_share(f'displayed contribution L{i} [0..255]', shown_i, np.clip(0.5 + 4 * contrib, 0, 1) * 255, 1.0)
    # phase semantics (semantics.wgsl, "Line / Edge" view): sem_l = angle(c_up^2 / |c_up| * conj(f)), f = level l, c = level l+1
    print('phase semantics (Line / Edge view)')
    if keep is not None:   # the view shows the repaired statistics before the slider blend (done in noise_band.wgsl)
        for i, unblended in enumerate(worm.rep_unblended):
            worm.y[i].update(unblended)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None]
    sems = {}
    for variant, flag in [('measured', 1), ('repaired', 0)]:
        key = 'phase' if variant == 'measured' else 'phase_rep'
        dkey = 'dir' if variant == 'measured' else 'dir_rep'
        for l in range(n_analysis - 1):
            out = gpu.texture(*sizes[l], 'rgba32float')
            gpu.run(P['sem'], [gpu.uniform(l, flag, int(ALIGN), 0), analysis[l], reps[l], analysis[l + 1], reps[l + 1], out], *sizes[l])
            sems[variant, l] = g = gpu.read(out)
            if l >= len(worm.y) - 1 or key not in worm.y[l + 1]:
                continue
            # (a) against the package: phu.get_phi_quick on the Worm's phases (only where level l+1 has energy)
            c = torch.polar(worm.y[l + 1]['G-H-complex'].abs(), worm.y[l + 1][key])
            f = torch.polar(torch.ones_like(worm.y[l][key]), worm.y[l][key])
            ref = (phu.get_phi_quick_aligned(c, f, worm.y[l + 1][dkey], worm.y[l][dkey]) if ALIGN
                   else phu.get_phi_quick(c, f, upscaling='bicubic'))
            strong = (ref.abs() > 0.05)[0, 0].numpy()
            ok &= report_share(f'semantics L{l} {variant} [rad]', g[..., 0][strong], ref.angle()[0, 0].numpy()[strong], 1e-2, period=2 * np.pi)
    # (b) where level l is fully extrapolated, its semantics are the repair's phi of level l+1, only upsampled
    #     (phu.upscale_complex_phase, as in repair_level.wgsl): the semantics map stays the same down the levels
    for l in range(min(cut, n_analysis - 1)):
        if l + 2 >= n_analysis:
            continue
        phi_c = gpu.read(phis[l + 1])
        phi_t = torch.complex(to_t(phi_c[..., 0]), to_t(phi_c[..., 1]))
        if ALIGN:   # as in phu.extrapolate_phi_aligned: the sign of the angle upsampled in the convention of level l
            ang_abs_up = isu.upsample(phi_t.angle().abs(), sizes[l][::-1], mode='bicubic')
            sgn = torch.sign(phu.upsample_aligned(phi_t.angle(), worm.y[l + 1]['dir_rep'], worm.y[l]['dir_rep'], sizes[l][::-1]))
            phi_up = torch.polar(isu.upsample(phi_t.abs(), sizes[l][::-1], mode='bicubic'), ang_abs_up * sgn)
        else:
            phi_up = phu.upscale_complex_phase(phi_t, sizes[l][::-1], mode='bicubic')
        extrap = (worm.a[l]['mask'] == 0)[0, 0].numpy() & (phi_up.abs() > 1e-3)[0, 0].numpy()
        if extrap.any():
            ok &= report_share(f'semantics L{l} = upsampled L{l + 1} ({100 * extrap.mean():.0f}% px)', sems['repaired', l][..., 0][extrap],
                               phi_up.angle()[0, 0].numpy()[extrap], 1e-2, period=2 * np.pi)
    # display modes 19 / 20 from the semantics texture
    def structure_ref(phi, t):
        line, edge, none = np.array([0.357, 0.439, 0.533]), np.array([0.749, 0.047, 0.298]), np.array([0.945, 0.937, 0.890])
        hue = line + (edge - line) * (np.sin(phi) ** 2)[..., None]
        return (none + (hue - none) * np.clip(2 * t, 0, 1)[..., None]) * 255
    for mode, variant in [(19, 'measured'), (20, 'repaired')]:
        tex0 = gpu.texture(*sizes[0], 'rgba32float')
        gpu.run(P['sem'], [gpu.uniform(0, int(variant == 'measured'), int(ALIGN), 0), analysis[0], reps[0], analysis[1], reps[1], tex0], *sizes[0])
        s0 = sems[variant, 0]
        tmp, disp = gpu.texture(*sizes[0], 'rgba32float'), gpu.texture(*sizes[0], 'rgba32float')   # display form (sem_smooth.wgsl)
        gpu.run(P['semH'], [tex0, tmp], *sizes[0]); gpu.run(P['semV'], [tmp, disp], *sizes[0])
        ok &= report_share(f'displayed semantics, {variant} [0..255]', render_view(mode, aux=disp, aux_scale=1.5),
                           structure_ref(s0[..., 0], 1.5 * s0[..., 1]), 1.0)

    if analyse_foveated:   # information only: the same pipeline analysing Python's float32 img.fov
        R32 = reference(img_path, vc, R['img'].fov, intensity_fit)
        d = (torch.from_numpy(g_final[..., 3].copy()) - R32['noise'][0, 0]).abs()
        print(f'  (info: noise from the GPU vs. Python analysing its float32 img.fov: mean |diff| {d.mean():.2e}, '
              f'max {d.max():.2e}; noise std {R32["noise"].std():.2e})')

    print('\nAll stages match the Python package.' if ok else '\nSome stages do NOT match.')
    return ok


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else os.path.join(HERE, '..', 'images', 'face.jpg')
    uniform = next((float(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--uniform=')), None)
    fit = 'cutoff' if '--fit=cutoff' in sys.argv else 'weighted'
    keep = next((tuple(float(v) for v in a.split('=')[1].split(',')) for a in sys.argv[1:] if a.startswith('--keep=')), None)
    sys.exit(0 if main(path, uniform, '--analyse-foveated' in sys.argv, fit, keep, '--measured' in sys.argv) else 1)
