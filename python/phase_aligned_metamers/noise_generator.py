import operator

import torch
import torch.nn.functional as F

import phase_aligned_metamers.phase_analysis_utils as phu
import phase_aligned_metamers.imagestats_utils as isu


class NoiseGenerator:
    """
    Sparse convolution noise generator.

    Impulses lie on a grid with a spacing of 8 px. Each (row, col) entry of `impulse_offsets` places one such
    grid at that offset within the 8x8 cell, and the noise band is the sum over all offsets. With all 64 offsets
    (default) every pixel carries an impulse, i.e. the densest noise. Fewer offsets give a sparser noise and a
    faster synthesis.
    """

    def __init__(self, size, impulse_offsets=None, device=None):
        """
        Args:
            size: [N, C, H, W] size of the noise, as in the image pyramid.
            impulse_offsets: list of (row, col) offsets within the 8x8 impulse cell, each in 0..7.
                Default None uses all 64 offsets, [(r, c) for r in range(8) for c in range(8)].
            device: torch device; defaults to CUDA if available, otherwise CPU.

        Raises:
            ValueError: if impulse_offsets is empty or contains an offset outside 0..7.
        """
        self.size = size
        if impulse_offsets is None:
            impulse_offsets = [(r, c) for r in range(8) for c in range(8)]
        self.impulse_offsets = self.validate_impulse_offsets(impulse_offsets)
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        self.impulse_maps = self.generate_impulse_map((1,1,self.size[2],self.size[3]), self.impulse_offsets, device=self.device)
        self.impulse_maps = F.pad(self.impulse_maps, pad=(8,8,8,8))

        self.filtered_maps = self.generate_prefiltered_maps(self.impulse_maps, device=self.device)

        self.noises = []
        self.noises_small = []

    def generate_noise(self, pyr, cutoff, keep_intensity=1.0, keep_dir=1.0, keep_phase=1.0, seed=42):
        """
        Synthesizes one noise band per scale below `cutoff` from the repaired statistics
        ('intensity_rep', 'dir_rep', 'phase_rep') and stores it in pyr[scale]['noise_rep'] / ['noise_rep_up'].
        Results are also collected in self.noises_small (at the level's size) and self.noises (full resolution).
        Each call starts from empty results, so consecutive calls are independent.

        keep_intensity, keep_dir, keep_phase (each in [0, 1]) control how much of the original statistic
        is used: 1 = original (default), 0 = fully random, in between = blend between the two.
        The random values are drawn from `seed`, so the same seed gives the same random fields; sweeping
        the keep_* values therefore smoothly interpolates between one fixed random noise and the original.
        The seed is only used if one of the keep_* values is < 1.
        """
        self.noises = []
        self.noises_small = []

        randomize = min(keep_intensity, keep_dir, keep_phase) < 1.0
        if randomize:
            # CPU generator: same random numbers as torch.manual_seed(seed) + torch.rand(...) used originally
            generator = torch.Generator(device='cpu').manual_seed(seed)

        for scale in range(cutoff):
            intensity = pyr[scale]['intensity_rep']
            dir = pyr[scale]['dir_rep']
            phase = pyr[scale]['phase_rep']

            if randomize:
                intensity, dir, phase = self.randomize_stats(intensity, dir, phase, keep_intensity, keep_dir, keep_phase, generator)

            noise = self.generate_noise_band(dir.to(self.device),
                                             phase.to(self.device),
                                             intensity.to(self.device),
                                             self.filtered_maps,
                                             self.impulse_offsets,
                                             pad_size=8,
                                             device=self.device)
            self.noises_small.append(noise)

            noise_up = isu.upsample(noise, size=pyr[0]['img'].shape[2:])

            pyr[scale]['noise_rep'] = noise / noise.std()
            pyr[scale]['noise_rep_up'] = noise_up / noise_up.std()
            pyr[scale]['noise_rep_up'] *= pyr[scale]['intensity_modulator']

            self.noises.append(pyr[scale]['noise_rep_up'])
        return pyr

    @staticmethod
    def randomize_stats(intensity, dir, phase, keep_intensity, keep_dir, keep_phase, generator):
        """
        Blends each statistic map with a random one: keep * original + (1 - keep) * random.
        Angles are blended on the unit circle (direction in the doubled-angle domain, since it is only defined modulo pi).
        Random values are always drawn in the same order (intensity, dir, phase), so results only depend on the seed.
        """
        shape, device = intensity.shape, intensity.device
        intensity_random = torch.rand(shape, generator=generator).to(device) * intensity.std()
        dir_random = torch.rand(shape, generator=generator).to(device) * torch.pi - torch.pi           # doubled below -> covers the full circle
        phase_random = torch.rand(shape, generator=generator).to(device) * 2 * torch.pi - torch.pi

        def cyclical_blend(original, random, keep):
            blended = keep * torch.polar(torch.ones_like(original), original) + (1 - keep) * torch.polar(torch.ones_like(random), random)
            return blended.angle()

        intensity = keep_intensity * intensity + (1 - keep_intensity) * intensity_random
        dir = cyclical_blend(dir * 2, dir_random * 2, keep_dir) / 2
        phase = cyclical_blend(phase, phase_random, keep_phase)
        return intensity, dir, phase

    def get_noise_enhancement(self, a, cutoff):
        """Sum of the generated noise bands, each faded out where its pyramid level is still resolved (a = Worm.a masks)."""
        noise = torch.zeros_like(a[0]['mask_FR'])  # stays zero if no layer is synthesized (cutoff 0)
        for scale in range(cutoff):
            noise = noise + self.noises[scale] * (1 - a[scale]['mask_FR'])
        return noise


    @staticmethod
    def validate_impulse_offsets(impulse_offsets):
        """Returns impulse_offsets as a list of (row, col) int tuples; raises ValueError if invalid."""
        offsets = []
        for offset in impulse_offsets:
            try:
                row, col = offset
                row, col = operator.index(row), operator.index(col)
            except (TypeError, ValueError):
                raise ValueError(f"impulse offset {offset!r} must be a (row, col) pair of integers") from None
            if not (0 <= row <= 7 and 0 <= col <= 7):
                raise ValueError(f"impulse offset {offset!r} must have row and col in 0..7")
            offsets.append((row, col))
        if len(offsets) == 0:
            raise ValueError("impulse_offsets must contain at least one (row, col) offset")
        return offsets

    @staticmethod
    def generate_impulse_map(size, impulse_offsets, device=None):

        device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # one impulse every 8 px, starting at the first offset (row, col); set in a single operation
        # (setting them one by one launches one GPU kernel per impulse, which takes seconds on CUDA)
        impulses = torch.zeros(size, device=device)
        row, col = impulse_offsets[0]
        impulses[0, 0, row::8, col::8] = 1.0
        return impulses

    @staticmethod
    def generate_prefiltered_maps(impulse_maps, device=None):
        device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        filters = phu.get_filters()
        filters = [f[:,:,1:-1,1:-1].to(device) for f in filters]

        pre_filtered_maps=[]
        for filter in filters:
            map = F.conv2d(impulse_maps.to(device), filter, stride=1, padding=3, groups=1)
            pre_filtered_maps.append(map)
        tensor = torch.cat(pre_filtered_maps, dim=0)

        return tensor

    @staticmethod
    def generate_noise_band(dir_field, phase_field, var_field, filtered_maps_org, impulse_offsets, pad_size, device=None):
        current_size = dir_field.shape[2:]
        noises = []

        device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        fields = torch.cat([dir_field, phase_field, var_field], dim=1).to(device)

        for H_shift, W_shift in impulse_offsets:
            # statistics at the impulses of this grid offset, held constant over the 7x7 filter support around each impulse
            fields_filtered_small = fields[:,:,H_shift::8, W_shift::8]
            fields_filtered_upscaled = F.interpolate(fields_filtered_small, size=current_size, mode='nearest')
            fields_filtered = torch.roll(fields_filtered_upscaled, shifts=(-3+H_shift, -3+W_shift), dims=(2, 3))

            filtered_maps = filtered_maps_org[:,:,pad_size-H_shift:,pad_size-W_shift:]
            filtered_maps = filtered_maps[:,:,:current_size[0],:current_size[1]]

            dir_filtered = fields_filtered[:,0:1,:,:]
            dir_cos = torch.cos(dir_filtered).to(device)
            dir_sin = torch.sin(dir_filtered).to(device)

            ka = dir_cos**2
            kb = -2 * dir_cos * dir_sin
            kc = dir_sin**2
            G_steered = (ka * filtered_maps[0].unsqueeze(0) + kb * filtered_maps[1].unsqueeze(0) + kc * filtered_maps[2].unsqueeze(0))


            ka = dir_cos**3
            kb = -3 * dir_cos**2 * dir_sin
            kc = 3 * dir_cos * dir_sin**2
            kd = -dir_sin**3
            H_steered = (ka * filtered_maps[3].unsqueeze(0) + kb * filtered_maps[4].unsqueeze(0) + kc * filtered_maps[5].unsqueeze(0) + kd * filtered_maps[6].unsqueeze(0))


            phase_filtered = fields_filtered[:,1:2,:,:]
            phase_adjusted = torch.cos(phase_filtered).to(device)  * G_steered - torch.sin(phase_filtered).to(device)  * H_steered

            var_filtered = fields_filtered[:,2:3,:,:]
            phase_adjusted = phase_adjusted * var_filtered
            noises.append(phase_adjusted)
        return sum(noises)