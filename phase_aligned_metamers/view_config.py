import torch
import phase_aligned_metamers.imagestats_utils as isu

class ViewConfig:
    def __init__(self, device, screen_width_cm:float, screen_width_px:float, img_size, viewing_distance:float,
                 gaze_pos_px=None, viewer_pos_cm=None, uniform_level=None, _debug=False):
        """
        Viewing condition: display pixel pitch, viewer position and gaze position. From these, computes per image pixel
        the eccentricity (ecc_map, degrees), the pyramid level (pyr_level_map) and the equivalent Gaussian blur
        sigma (kernel_map, px).

        The image is assumed to be centred on the screen; only the display's pixel pitch matters
        (screen_width_cm / screen_width_px), not the full screen size or resolution.

        Coordinates: (vertical, horizontal) in image pixels or cm, vertical pointing down, horizontal pointing right;
        cm coordinates are relative to the image centre.

        gaze_pos_px:   (row, col) of the gaze point in image pixels. Default: image centre.
        viewer_pos_cm: (vertical, horizontal, distance) of the viewer's eye in cm, relative to the image centre;
                       distance > 0 is in front of the screen. Default: (0, 0, viewing_distance), i.e. centred.
                       If given, it overrides viewing_distance.
        uniform_level: if given (e.g. 2.5), the whole image uses this pyramid level instead of the
                       eccentricity-dependent one, i.e. the image is foveated uniformly. See set_uniform_level().

        Use update(...) to change gaze or viewer position later; it recomputes all maps. Image and Worm read the
        maps when they are created, so rebuild them (or call Image.update_foveation) after an update.
        """
        self.DEVICE = device

        # screen information
        self.px_per_cm = screen_width_px / screen_width_cm  # display's pixel pitch

        # image resolution [N, C, H, W], centre pixel and size on the screen in cm
        self.img_size = list(img_size)
        self.img_center_px = (self.img_size[2] // 2, self.img_size[3] // 2)
        self.img_size_cm = (self.img_size[2] / self.px_per_cm, self.img_size[3] / self.px_per_cm)

        # viewer and gaze (defaults: centred viewer at viewing_distance, gaze at the image centre)
        self.viewer_pos_cm = tuple(viewer_pos_cm) if viewer_pos_cm is not None else (0.0, 0.0, viewing_distance)
        self.gaze_pos_px = tuple(gaze_pos_px) if gaze_pos_px is not None else self.img_center_px
        self.uniform_level = uniform_level

        self.ecc_map = None        # eccentricity per pixel [deg]
        self.pyr_level_map = None  # pyramid level per pixel
        self.kernel_map = None     # equivalent Gaussian blur sigma per pixel [px]
        self.recompute_maps()

        if _debug:
            self.print_info()

    @property
    def viewing_distance(self):
        return self.viewer_pos_cm[2]

    def print_info(self):
        print("ViewConfig:")
        print(f"  - Image size:              {self.img_size}")
        print(f"  - Image size (H x W):      {self.img_size_cm[0]:.2f} x {self.img_size_cm[1]:.2f} cm")
        print(f"  - Pixel pitch:             {self.px_per_cm} px/cm")
        print(f"  - Viewer position (cm):    vertical {self.viewer_pos_cm[0]}, horizontal {self.viewer_pos_cm[1]}, distance {self.viewer_pos_cm[2]}")
        print(f"  - Gaze position (px):      {self.gaze_pos_px}")
        if self.uniform_level is not None:
            print(f"  - Uniform pyramid level:   {self.uniform_level}")
        print(f"  - Eccentricity range:      {self.ecc_map.min().item():.2f} - {self.ecc_map.max().item():.2f} deg")
        print(f"  - Pyramid level range:     {self.pyr_level_map.min().item():.2f} - {self.pyr_level_map.max().item():.2f}")

    def update(self, gaze_pos_px=None, viewer_pos_cm=None, _debug=False):
        """
        Changes the gaze position (row, col in px) and/or the viewer position (vertical, horizontal, distance in cm)
        and recomputes ecc_map, pyr_level_map and kernel_map. Arguments that are None keep their current value.
        Afterwards, rebuild the Worm and call Image.update_foveation(view_config.kernel_map).
        """
        if gaze_pos_px is not None:
            self.gaze_pos_px = tuple(gaze_pos_px)
        if viewer_pos_cm is not None:
            self.viewer_pos_cm = tuple(viewer_pos_cm)
        self.recompute_maps()
        if _debug:
            self.print_info()

    def recompute_maps(self):
        self.ecc_map = self.compute_eccentricity_map()
        if self.uniform_level is not None:
            self.pyr_level_map = torch.full_like(self.ecc_map, float(self.uniform_level))
        else:
            self.pyr_level_map = self.map_eccentricity_to_pyr_level(self.ecc_map)  # linear, as used for the paper
        # Gaussian blur sigma [px] per pixel that is equivalent to the Gaussian pyramid down to the respective level
        # (0 in the fovea, i.e. no blur). See isu.pyramid_level_to_sigma.
        self.kernel_map = isu.pyramid_level_to_sigma(self.pyr_level_map)

    def set_uniform_level(self, level):
        """
        Uses one pyramid level for the whole image instead of the eccentricity-dependent one (None switches back).
        ecc_map is left untouched. Create Image and Worm *after* calling this, since they read these maps on construction.
        """
        self.uniform_level = level
        self.recompute_maps()

    def px_to_cm(self, row, col):
        """Pixel (row, col) -> position in cm relative to the image centre, using pixel centres."""
        return ((row - (self.img_size[2] - 1) / 2) / self.px_per_cm,
                (col - (self.img_size[3] - 1) / 2) / self.px_per_cm)

    def compute_eccentricity_map(self):
        """
        Eccentricity per image pixel in degrees: the angle at the viewer's eye between the direction to the gaze
        point and the direction to the pixel. Screen plane: z = 0; viewer at z = -distance.
        """
        dev = self.DEVICE
        viewer = torch.tensor((self.viewer_pos_cm[0], self.viewer_pos_cm[1], -self.viewer_pos_cm[2]), dtype=torch.float32, device=dev)

        # 3D positions of all pixel centres (H, W, 3) and of the gaze point, in cm
        rows = torch.arange(self.img_size[2], dtype=torch.float32, device=dev)
        cols = torch.arange(self.img_size[3], dtype=torch.float32, device=dev)
        Y, X = torch.meshgrid(*self.px_to_cm(rows, cols), indexing='ij')
        pixels = torch.stack((Y, X, torch.zeros_like(X)), dim=-1)
        gaze = torch.tensor((*self.px_to_cm(*self.gaze_pos_px), 0.0), dtype=torch.float32, device=dev)

        to_pixel = pixels - viewer
        to_gaze = gaze - viewer
        cos_angle = (to_pixel * to_gaze).sum(dim=-1) / (to_pixel.norm(dim=-1) * to_gaze.norm() + 1e-8)
        angle_deg = torch.rad2deg(torch.acos(cos_angle.clamp(-1.0, 1.0)))
        return angle_deg.unsqueeze(0).unsqueeze(0)

    def map_eccentricity_to_pyr_level(self, x):
        """
        Maps eccentricity values to 'level' values using a piecewise linear function.
        The levels correspond to the Gaussian pyramid levels that we can use safely per respective eccentricity.
        The threshold values were determined in the calibration study.
        """

        if isinstance(x, (int, float)):
            x = torch.tensor([x], dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        y = torch.zeros_like(x, dtype=torch.float32)

        # Segment 1: 0 to 4, y = 0
        mask1 = (x >= 0) & (x < 4)
        y[mask1] = 0

        # Segment 2: 4 to 8, y grows from 0 to 2
        mask2 = (x >= 4) & (x < 8)
        y[mask2] = 2 * (x[mask2] - 4) / (8 - 4)

        # Segment 3: 8 to 16, y grows from 2 to 3
        mask3 = (x >= 8) & (x < 16)
        y[mask3] = 2 + 1 * (x[mask3] - 8) / (16 - 8)

        # Segment 4: 16 to 24, y grows from 3 to 4
        mask4 = (x >= 16) & (x < 24)
        y[mask4] = 3 + 1 * (x[mask4] - 16) / (24 - 16)

        # Segment 5: 24 to Inf, y = 4
        mask5 = x >= 24
        y[mask5] = 4

        return y.squeeze() if y.numel() == 1 else y

    def map_eccentricity_to_pyr_level_squared(self, x):
        """
        Alternative (currently unused) mapping: same breakpoints as map_eccentricity_to_pyr_level
        (0 / 2 / 3 / 4 at 4 / 8 / 16 / 24 deg), but interpolated linearly in level**2 instead of level,
        i.e. level = sqrt(piecewise linear interpolation of level**2). Between the breakpoints this gives
        higher levels (more blur) than the linear mapping.
        """
        if isinstance(x, (int, float)):
            x = torch.tensor([x], dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(dtype=torch.float32)
        y_pow = torch.zeros_like(x, dtype=torch.float32)  # level**2

        # Segment 1: 0 to 4 deg: level 0
        mask1 = (x >= 0) & (x < 4)
        y_pow[mask1] = 0.0

        # Segment 2: 4 to 8 deg: level**2 from 0 to 2**2
        mask2 = (x >= 4) & (x < 8)
        y_pow[mask2] = 0**2 + (2**2 - 0**2) * (x[mask2] - 4.0) / (8.0 - 4.0)

        # Segment 3: 8 to 16 deg: level**2 from 2**2 to 3**2
        mask3 = (x >= 8) & (x < 16)
        y_pow[mask3] = 2**2 + (3**2 - 2**2) * (x[mask3] - 8.0) / (16.0 - 8.0)

        # Segment 4: 16 to 24 deg: level**2 from 3**2 to 4**2
        mask4 = (x >= 16) & (x < 24)
        y_pow[mask4] = 3**2 + (4**2 - 3**2) * (x[mask4] - 16.0) / (24.0 - 16.0)

        # Segment 5: 24+ deg: level 4
        mask5 = x >= 24
        y_pow[mask5] = 16.0

        y = torch.sqrt(y_pow)  # back from level**2 to level

        return y.squeeze() if y.numel() == 1 else y
