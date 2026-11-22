import torch
import torch.nn.functional as F
import numpy as np

import phase_aligned_metamers.imagestats_utils as isu
import phase_aligned_metamers.phase_analysis_utils as phu

class Worm:
    MIN_LEVEL_SIZE = 8  # smallest allowed image side [px] of a pyramid level (9x9 steerable filters, reflect padding)

    def __init__(self, device, img, pyr_level_map, MAX_DEPTH, fit_levels=None, _debug=False):
        """
        fit_levels:    number of pyramid levels above the cutoff level that are built for the intensity fit. None (default):
                       as many as possible, i.e. down to the smallest level with at least MIN_LEVEL_SIZE px per side
                       (as in the paper's code, where the pyramid was 9 levels deep).
        """
        self.DEVICE = device

        self.img = img.clone()

        self.cutoff = None                  # is max. 4 by design. pyr_level_map we input for its computation is [0,4] max.
        self.pyr_level_map = pyr_level_map  # [0,4] max
        self.depth = None                   # is max. 5 by design; cutoff + 1
        self.image_pyr_overhead = None      # = fit_levels (after capping): additional layers of Laplace (and therefore
                                            # Gaussian) pyramid for intensity extrapolation
        self.analysis_pyr_overhead = 1      # Overhead for analysis computations. Analysis should provide 1 extra level
                                            # above cutoff for phase computations.

        self.size = img.shape[2:]

        self.y    = None  # list[dicts], with keys 'img', 'laplace', 'R', 'theta', ...
        self.cb   = None  # list[dicts]
        self.cr   = None  # list[dicts]
        self.a    = None  # list[dicts], carries masks and rings
        self.meta = None  # list[dicts], carries size and other metadata per layer

        phu.init_phu()

        # initialize masks, calculate necessary Gaussian pyramid depth
        self.a, self.cutoff = self.compute_cutoff_and_masks(MAX_DEPTH)
        self.depth = self.cutoff + 1
        self.image_pyr_overhead = self.cap_fit_levels(fit_levels, _debug)


        # initialize Y channel maps
        self.y = self.init_image_pyramid(self.img[:,0:1,...])
        self.meta = self.compute_size_per_layer(self.y)
        self.y = self.compute_laplace_layers(self.y)


        # finish mask and ring calculations
        self.a = self.compute_small_masks(self.a)
        self.a = self.compute_rings(self.a)


        # if image has color channels:
        if self.img.shape[1] > 1:
            self.cb = self.init_image_pyramid(self.img[:,1:2,...])
            self.cr = self.init_image_pyramid(self.img[:,2:3,...])



        if _debug:
            print("Worm initialized with:")
            print(f"  - Image size:              {self.size}")
            print(f"  - Cutoff layer:            {self.cutoff}")
            print(f"  - Pyramid depth:           {self.depth}")
            print(f"  - Levels for intensity fit:{self.image_pyr_overhead:2d} (levels {self.cutoff}..{self.cutoff + self.image_pyr_overhead - 1})")
            print(f"  - Y channel layers:        {len(self.y)}")
            if self.cb is not None:
                print(f"  - Cb channel layers:       {len(self.cb)}")
                print(f"  - Cr channel layers:       {len(self.cr)}")
            else:
                print("  - Grayscale image: no Cb/Cr pyramids")



    def cap_fit_levels(self, fit_levels, _debug=False):
        """Reduces fit_levels if the pyramid would get smaller than MIN_LEVEL_SIZE px per side (None: as many as possible)."""
        n_levels_possible = int(np.floor(np.log2(min(self.size) / self.MIN_LEVEL_SIZE))) + 1  # levels 0 .. n-1
        if fit_levels is None:
            fit_levels = n_levels_possible - self.depth
        capped = min(fit_levels, n_levels_possible - self.depth)
        if capped < 2:
            raise ValueError(f"Image too small: only {capped} level(s) left for the intensity fit after the cutoff (need >= 2).")
        if capped < fit_levels and _debug:
            print(f"fit_levels reduced from {fit_levels} to {capped}: smallest pyramid level reached.")
        return capped

    # Operations on self.a
    def compute_cutoff_and_masks(self, MAX_DEPTH):
        """Computes the cutoff, which is the maximum pyramid layer we need according to our viewing conditions."""

        total_pixels = self.pyr_level_map.numel()
        a = []
        cutoff = None

        for i in range(MAX_DEPTH):
            mask = (self.pyr_level_map-(i)).clamp(min=0, max=1)
            mask = 1 - mask
            a.append({'mask_FR': mask})  # full resolution mask per layer with soft transition

            mask = (mask >= 1).float()
            mask_pixels = mask.sum().item()
            if mask_pixels == total_pixels:
                cutoff = i  # i marks the highest pyramid level we need.
                break

        if cutoff is None:
            raise ValueError(f"pyr_level_map reaches level {self.pyr_level_map.max().item():.2f}, but "
                             f"MAX_DEPTH={MAX_DEPTH} only supports levels 0 to {MAX_DEPTH - 1}.")
        return a, cutoff
    def compute_small_masks(self, a):
        for layer_a, layer_meta in zip(a, self.meta):
            layer_a['mask'] = F.interpolate(layer_a['mask_FR'], size=layer_meta['size'], mode='nearest')
        return a
    def compute_rings(self, a):
        a[0]['ring'] = a[0]['mask_FR']
        a[0]['ring_small'] = a[0]['mask']
        for i in range(1, len(a)):
            a[i]['ring'] = a[i]['mask_FR'] - a[i-1]['mask_FR']
            a[i]['ring_small'] = F.interpolate(a[i]['ring'], size=self.meta[i]['size'], mode='nearest')
        return a



    # Operations on self.meta
    def compute_size_per_layer(self, pyr):
        return [{'size': layer['img'].shape[2:]} for layer in pyr]


    # Setting up the image pyramids
    def init_image_pyramid(self, img):
        """Image pyramid as a list of dictionaries, one per level, with the Gaussian pyramid level as 'img'."""
        mips = isu.get_gauss_mipmaps(img.to(self.DEVICE), self.depth + self.image_pyr_overhead)
        return [{'img': mip} for mip in mips]

    def compute_laplace_layers(self, pyr):
        for i in range(self.depth + self.image_pyr_overhead - 1):
            laplace = pyr[i]['img'] - isu.upsample(pyr[i + 1]['img'], size=pyr[i]['img'].shape[2:])
            pyr[i]['laplace'] = laplace  # at the level's own size; full resolution: isu.upsample(laplace, size)
        return pyr
    def compute_foveated_imgs(self):
        """
        Foveation by blending the Gaussian pyramid: every level is upsampled (bicubic) to full resolution and weighted
        with its full-resolution ring. The rings are differences of consecutive mask_FR and therefore sum to exactly 1.
        (Weighting at level resolution and upsampling afterwards breaks that: the weights then sum to ~0.88-1.09,
        which scales the image locally and tints the chroma channels.)
        """
        def blend(pyr):
            return sum([isu.upsample(pyr[scale]['img'], size=self.meta[0]['size']) * self.a[scale]['ring']
                        for scale in range(self.depth)])

        self.y[0]['fov'] = blend(self.y)
        if self.cb is not None:
            self.cb[0]['fov'] = blend(self.cb)
        if self.cr is not None:
            self.cr[0]['fov'] = blend(self.cr)
        return


    # Analysis operations
    def compute_R(self, pyr):
        for layer in pyr[:self.depth + self.analysis_pyr_overhead]:  # we need to compute R for all layers that we use for analysis, which are the layers up to cutoff + 1 (for phase extrapolation) and therefore also the layers up to depth + 1 (because depth = cutoff)
            layer['R'] = phu.get_basis_images(layer['img'])
        return pyr
    def compute_theta(self, pyr):
        for layer in pyr[:self.depth + self.analysis_pyr_overhead]:
            theta, S = phu.get_theta_d(layer['R'], blur_kernel_size=5)
            layer['dir'] = theta
            layer['S'] = S
        return pyr
    def compute_steered_G_H(self, pyr):
        for layer in pyr[:self.depth + self.analysis_pyr_overhead]:

            G_ka, G_kb, G_kc = phu.get_Gk(layer['dir'])
            G = (G_ka * layer['R'][0] + G_kb * layer['R'][1] + G_kc * layer['R'][2])

            H_ka, H_kb, H_kc, H_kd = phu.get_Hk(layer['dir'])
            H = (H_ka * layer['R'][3] + H_kb * layer['R'][4] + H_kc * layer['R'][5] + H_kd * layer['R'][6])

            layer['G-H-complex'] = torch.complex(G, H)
        return pyr
    def compute_phase(self, pyr):
        for layer in pyr[:self.depth + self.analysis_pyr_overhead]:
            layer['phase'] = layer['G-H-complex'].angle()  # equivalent to:  atan2(H, G)
        return pyr
    def compute_intensity_modulators(self, pyr, _debug=False):
        """
        Noise amplitude per level (as in the paper): a least-squares line std = m * level + b through the std of the
        Laplacian bands from the cutoff level on, extrapolated to the levels below the cutoff and clamped at 0.
        (The web version additionally offers a weighted fit over all bands, which avoids the jump of this fit when
        the cutoff level changes; see web/shaders/fit.wgsl.)
        """
        x = np.arange(self.cutoff, len(pyr) - 1, dtype=np.float64)
        y = np.array([pyr[int(i)]['laplace'].std().item() for i in x])   # std at the level's own size (~= full-res std)
        A = np.stack((x, np.ones_like(x)), axis=1)
        m, b = np.linalg.solve(A.T @ A, A.T @ y)

        for i, layer in enumerate(pyr[:self.depth + self.analysis_pyr_overhead]):
            layer['intensity_modulator'] = max(m * i + b, 0.0)  # a negative amplitude would invert the noise band
            if _debug and m * i + b < 0:
                print(f'intensity_modulator of level {i} would be negative ({m * i + b:.4f}); clamped to 0.')
            layer['intensity'] = layer['G-H-complex'].abs() * layer['intensity_modulator']
        self.fit = {'levels': x, 'std': y, 'weights': np.ones_like(x), 'm': m, 'b': b}

        if _debug:
            self.plot_intensity_fit(pyr, x, y, m, b)
        return pyr

    def plot_intensity_fit(self, pyr, x, y, m, b, w=None):
        """Debug plot: Laplacian std per level, the points of the fit (size = weight), and the extrapolated modulators."""
        import matplotlib.pyplot as plt
        all_levels = np.arange(len(pyr) - 1)
        all_std = [pyr[i]['laplace'].std().item() for i in all_levels]
        used = np.arange(self.depth + self.analysis_pyr_overhead)

        teal, magenta, orange = '#377d8e', '#be0b4b', '#de8400'
        fg = isu.THEME_COLOR   # axes, labels and neutral marks follow the notebook theme (light/dark)
        modulators = np.maximum(m * used + b, 0)
        x_max = all_levels[-1] + 0.5

        # small text and thin strokes, to match the captions of isu.show* (~11-13 px in the notebook)
        with plt.rc_context({'font.size': 5.6, 'axes.titlesize': 5.6, 'axes.labelsize': 5.95, 'xtick.labelsize': 5.25,
                             'ytick.labelsize': 5.25, 'legend.fontsize': 5.25,
                             'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                             'axes.edgecolor': fg, 'axes.labelcolor': fg, 'xtick.color': fg, 'ytick.color': fg}):
            fig, ax = plt.subplots(figsize=(5.6, 3.0))
            ax.set_facecolor('none')
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
            ax.axvspan(-0.5, self.cutoff - 0.5, color=fg, alpha=0.06, lw=0)   # levels that get noise
            line_x = np.array([-0.5, x_max])
            ax.plot(line_x, m * line_x + b, color=teal, lw=0.9, zorder=2)
            ax.plot(all_levels, all_std, 'o', ms=3.2, mfc='none', mec=fg, mew=0.7, zorder=3)
            w = np.ones_like(x) if w is None else w
            ax.scatter(x, y, s=6 + 24 * w / w.max(), color=orange, zorder=4, linewidths=0)
            ax.plot(used[:self.cutoff], modulators[:self.cutoff], 'D', ms=3.2, color=magenta, mew=0, zorder=5)

            # labels at the data instead of a legend box
            ax.annotate('linear fit', (x_max, m * x_max + b), xytext=(4, 0), textcoords='offset points', color=teal, va='center', fontsize=5.25)
            ax.annotate('used for fit', (x[-1], y[-1]), xytext=(-6, 8), textcoords='offset points', color=orange, ha='right', fontsize=5.25)
            ax.annotate('measured std', (all_levels[0], all_std[0]), xytext=(8, 4), textcoords='offset points', color=fg, fontsize=5.25)
            if self.cutoff > 0:
                ax.annotate('noise amplitude', (used[0], modulators[0]), xytext=(8, -10), textcoords='offset points', color=magenta, fontsize=5.25)
                ax.text((self.cutoff - 1) / 2, 0.97, 'synthesized levels', transform=ax.get_xaxis_transform(),
                        ha='center', va='top', color=fg, alpha=0.7, fontsize=5.25)

            y_top = max(max(all_std), m * x_max + b) * 1.12
            ax.set_xlim(-0.5, x_max + 0.9)
            ax.set_ylim(0, y_top)
            ax.set_xticks(all_levels)
            ax.set_yticks([0, round(y_top, 2)])
            ax.tick_params(length=3)
            ax.set_xlabel('pyramid level')
            ax.set_ylabel('std of Laplacian band')
            isu.show_figure(fig, title='Intensity Fit', note=f'std = {m:.4f} \u00b7 level + {b:.4f}')

    def add_repaired_layer_proxies(self, pyr):
        for level in pyr[:self.depth + self.analysis_pyr_overhead]:
            level['intensity_rep'] = level['intensity']
            level['phase_rep'] = level['phase']
            level['dir_rep'] = level['dir']

        return pyr


    # Repairing Operations
    def repair_intensity(self, pyr):
        for scale in range(self.cutoff - 1, -1, -1):
            mask_fine   = self.a[scale]['mask']
            size_fine   = self.meta[scale]['size']

            intensity_fine   = pyr[scale]['intensity_rep']
            intensity_coarse = pyr[scale+1]['intensity_rep']

            intensity_coarse_up = isu.upsample(intensity_coarse, size=size_fine)

            intensity_repaired = intensity_fine * mask_fine + intensity_coarse_up * (1 - mask_fine)
            pyr[scale]['intensity_rep'] = intensity_repaired
        return pyr
    def repair_orientation(self, pyr):
        for scale in range(self.cutoff - 1, -1, -1):  # from cutoff-1 down to 0: the first repaired level is the one below the cutoff
            mask_fine   = self.a[scale]['mask']
            size_fine   = self.meta[scale]['size']

            coarse_dir = pyr[scale+1]['dir_rep']

            # upscale coarse_dir in [-pi, pi] and transform back to [-pi/2, pi/2]
            coarse_dir = torch.polar(torch.ones_like(coarse_dir), 2  *coarse_dir)  # displaying .angle() shows that it goes from -pi to pi
            coarse_dir_up = phu.upscale_complex_img(coarse_dir, size_up=size_fine, mode='bicubic')
            coarse_dir_up = torch.polar(coarse_dir_up.abs(), 0.5 * coarse_dir_up.angle())  # remap to -pi/2, pi/2

            fine_dir = pyr[scale]['dir']
            fine_dir = torch.polar(torch.ones_like(fine_dir), fine_dir)

            dir_repaired = fine_dir / fine_dir.abs() * mask_fine + coarse_dir_up / (coarse_dir_up.abs() + 1e-12) * (1 - mask_fine)
            pyr[scale]['dir_rep'] = dir_repaired.angle()
        return pyr
    def repair_phase(self, pyr):
        for scale in range(self.cutoff - 1, -1, -1):
            mask_fine   = self.a[scale]['mask']
            size_fine   = self.meta[scale]['size']

            phase_fine = torch.polar(torch.ones_like(pyr[scale]['phase']), pyr[scale]['phase'])

            # Only the angles are extrapolated; the magnitudes act as weights. Use the filter response magnitude |G + iH|
            # (not 'intensity', which is additionally scaled by the intensity modulator and can be 0).
            f_mag = pyr[scale+1]['G-H-complex'].abs()
            c_mag = pyr[scale+2]['G-H-complex'].abs()
            f = torch.polar(f_mag, pyr[scale+1]['phase_rep'])
            c = torch.polar(c_mag, pyr[scale+2]['phase_rep'])

            phi = phu.get_phi_quick(c, f, upscaling="bicubic")
            f_recon = phu.extrapolate_phi(phi, f, steps_up=1, upscaling_mode='bicubic')
            phase_up = phu.upscale_complex_img(f_recon, size_up=size_fine, mode='bicubic')

            phase_repaired = phase_fine / phase_fine.abs() * mask_fine + phase_up / (phase_up.abs() + 1e-12) * (1 - mask_fine)
            pyr[scale]['phase_rep'] = phase_repaired.angle()
        return pyr

