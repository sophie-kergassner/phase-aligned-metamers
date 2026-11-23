import torch
import phase_aligned_metamers.imagestats_utils as isu

class Image:
    def __init__(self, device, path, kernels, max_value=None, _debug=False):
        """
        Loads an image and prepares it for the pipeline.
        Color images (RGB/RGBA) are converted to YCbCr. Grayscale images (L/LA) are kept as a single
        channel, which then plays the role of Y; all pipeline steps only operate on Y anyway.
        max_value: see isu.import_image_NCHW (default: from the bit depth).
        """
        self.DEVICE = device

        self.path = path
        raw = isu.import_image_NCHW(f'{path}', max_value=max_value).to(self.DEVICE)

        self.is_gray = raw.shape[1] in (1, 2)  # grayscale, with or without alpha channel
        if self.is_gray:
            self.rgb = raw[:, 0:1, :, :]   # single channel
            self.ycbcr = self.rgb          # gray value == Y
        else:
            self.rgb = raw[:, :3, :, :]    # drop alpha if present
            self.ycbcr = isu.rgb_2_ycrcb(self.rgb)

        self.fov = self.compute_foveated_imgs_gauss(kernels)

        if _debug:
            print(f"Image loaded with shape {self.rgb.shape}" + (" (grayscale)" if self.is_gray else ""))


    def update_foveation(self, kernels):
        """Recomputes self.fov for a new blur map, e.g. after ViewConfig.update() changed the gaze or viewer position."""
        self.fov = self.compute_foveated_imgs_gauss(kernels)

    def compute_foveated_imgs_gauss(self, kernels):
        """Foveates every channel of self.ycbcr (Y, Cb, Cr -- or only Y for grayscale images)."""
        channels = [isu.nh_foveate_image(self.ycbcr[:, c:c+1, ...], kernels) for c in range(self.ycbcr.shape[1])]
        return torch.cat(channels, dim=1)
