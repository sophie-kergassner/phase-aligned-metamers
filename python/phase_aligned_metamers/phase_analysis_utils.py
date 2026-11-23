import torch
import torch.nn.functional as F

import phase_aligned_metamers.imagestats_utils as isu

filters = None
_filters_device = None
_filters_dtype = None
_blur_kernels = {}

def _ensure_filters_on(img):
    global filters, _filters_device, _filters_dtype
    target_device = img.device
    target_dtype = img.dtype
    if _filters_device != target_device or _filters_dtype != target_dtype:
        filters = [f.to(dtype=target_dtype, device=target_device) for f in filters]
        _filters_device = target_device
        _filters_dtype = target_dtype


def init_phu():
    global filters, _filters_device, _filters_dtype
    filters = get_filters()
    _filters_device = None
    _filters_dtype = None
    _blur_kernels.clear()


def get_filters(kernel_size=9, step=0.67) -> list:
    """
    Returns a list of 7 steerable filters [Ga, Gb, Gc, Ha, Hb, Hc, Hd] \n
    2nd derivative of Gaussian and their Hilbert transforms. \n
    [1, 1, kernel_size, kernel_size]
    """

    center = kernel_size // 2

    y = torch.linspace(-center * step, center * step, kernel_size)
    x = torch.linspace(-center * step, center * step, kernel_size)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    Ga = 0.9213 * (2*X**2 - 1) * torch.exp(-(X**2 + Y**2))
    Gb = 1.843 * X * Y * torch.exp(-(X**2 + Y**2))
    Gc = 0.9213 * (2*Y**2 - 1) * torch.exp(-(X**2 + Y**2))

    Ha = 0.9780 * (-2.254 * X + X**3) * torch.exp(-(X**2 + Y**2))
    Hb = 0.9780 * (-0.7515 + X**2) * (Y) * torch.exp(-(X**2 + Y**2))
    Hc = 0.9780 * (-0.7515 + Y**2) * (X) * torch.exp(-(X**2 + Y**2))
    Hd = 0.9780 * (-2.254 * Y + Y**3) * torch.exp(-(X**2 + Y**2))

    filter = [Ga, Gb, Gc, Ha, Hb, Hc, Hd]
    filter = [f.unsqueeze(0).unsqueeze(0) for f in filter]

    return filter

def get_basis_images(img):
    _ensure_filters_on(img)
    pad = filters[0].shape[2] // 2
    img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')

    R = [F.conv2d(img_pad, filter, padding=0) for filter in filters]

    return R

def get_C2(Ga, Gb, Gc, Ha, Hb, Hc, Hd):
    C2 = 0.5 * (Gb**2 - Gc**2) + 0.46875 * (Ha**2 - Hd**2) \
         + 0.28125 * (Hb**2 - Hc**2) + 0.1875 * (Ha * Hc - Hb * Hd)
    return C2

def get_C3(Ga, Gb, Gc, Ha, Hb, Hc, Hd):
    C3 = -Ga * Gb - Gb * Gc \
         -0.9375 * (Hc * Hd + Ha * Hb) - 1.6875 * Hb * Hc - 0.1875 * Ha * Hd
    return C3

def get_theta_d(R, blur_kernel_size=5):
    """
    Dominant orientation theta_d = atan2(C3, C2) / 2 and orientation strength S = |C2 + iC3|,
    with C2, C3 box-blurred (blur_kernel_size x blur_kernel_size).
    """

    C2 = get_C2(*R)
    C3 = get_C3(*R)

    key = (blur_kernel_size, C2.device, C2.dtype)
    if key not in _blur_kernels:
        _blur_kernels[key] = torch.ones((1, 1, blur_kernel_size, blur_kernel_size),
                                        device=C2.device, dtype=C2.dtype) / (blur_kernel_size * blur_kernel_size)
    kernel = _blur_kernels[key]
    pad = blur_kernel_size // 2
    C2 = F.conv2d(C2, kernel, padding=pad)
    C3 = F.conv2d(C3, kernel, padding=pad)

    theta_d = torch.atan2(C3, C2) / 2
    S = torch.sqrt(C2**2 + C3**2)
    return theta_d, S

def get_Gk(theta):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.float32)
    ka = torch.cos(theta)**2
    kb = -2 * torch.cos(theta) * torch.sin(theta)
    kc = torch.sin(theta)**2
    return ka, kb, kc

def get_Hk(theta):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.float32)
    ka = torch.cos(theta)**3
    kb = -3 * torch.cos(theta)**2 * torch.sin(theta)
    kc = 3 * torch.cos(theta) * torch.sin(theta)**2
    kd = -torch.sin(theta)**3
    return ka, kb, kc, kd


def upscale_complex_img(img, size_up, mode='bilinear'):
    # aligned with the [::2] pyramid sampling (see isu.upsample)
    return isu.upsample(img, size_up, mode=mode)

def get_phi_quick(c, f, upscaling="bilinear"):
    c_up = upscale_complex_img(c, size_up=f.shape[2:], mode=upscaling)

    c_hat = c_up**2 / (c_up.abs() + 1e-6)      # ĉ = c² / |c|
    phi   = c_hat * torch.conj(f)       # φ = ĉ f*

    return phi


def upscale_complex_phase(img, size_up, mode='bilinear'):
    phi_angle = img.angle()

    phi_angle_abs = abs(phi_angle)
    phi_angle_abs_up = isu.upsample(phi_angle_abs, size_up, mode=mode)
    phi_angle_up = phi_angle_abs_up * torch.sign(isu.upsample(phi_angle, size_up, mode=mode))
    phi_abs_up = isu.upsample(img.abs(), size_up, mode=mode)

    phi_up = torch.polar(phi_abs_up, phi_angle_up)
    return phi_up

def extrapolate_phi(phi, c, steps_up:int, upscaling_mode='bicubic'):
    size_current = c.shape[2:]
    size_up = 2**steps_up * size_current[0], 2**steps_up * size_current[1]

    phi_up = upscale_complex_phase(phi, size_up, mode=upscaling_mode)
    c_up = upscale_complex_img(c, size_up, mode=upscaling_mode)
    phase_scaling = 2**steps_up

    f_phase = phase_scaling * c_up.angle() - (phase_scaling -1 ) * phi_up.angle()
    f_phase = (f_phase + torch.pi) % (2 * torch.pi) - torch.pi
    f_recon = torch.polar(c_up.abs(), f_phase)

    return f_recon


# ---------------------------------------------------------------------------------------------------------------------
# Phase convention alignment (Worm(align_phase_convention=True); see fixes/1_phase_sign_alignment.md)
#
# The orientation is only defined modulo pi, but the steered H filter is odd: H(theta + pi) = -H(theta). So
# (dir, phase) and (dir + pi, -phase) describe the same local signal, and where dir wraps from -pi/2 to +pi/2 the
# phase is conjugated. Before complex responses of different pixels / levels are interpolated or combined, each is
# brought into the convention of a reference orientation: conjugated where its own orientation points the other way.
# ---------------------------------------------------------------------------------------------------------------------

def align_sign(theta, theta_ref):
    """-1 where orientation theta points away from theta_ref (cos(theta - theta_ref) < 0), else +1."""
    return torch.where(torch.cos(theta - theta_ref) < 0, -1.0, 1.0).to(theta.dtype)


def _cubic_weights(t, A=-0.75):
    """Cubic convolution weights of the 4 taps (base - 1 .. base + 2) for fractional position t, as grid_sample."""
    def k1(x): return ((A + 2) * x - (A + 3)) * x * x + 1
    def k2(x): return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
    return [k2(t + 1), k1(t), k1(1 - t), k2(2 - t)]


def upsample_aligned(x, theta, theta_ref, size):
    """
    isu.upsample(x, size, 'bicubic') with every one of the 16 taps aligned to the orientation of the target pixel:
    a tap at source pixel q is taken as-is where cos(theta[q] - theta_ref[p]) >= 0, otherwise conjugated (complex x)
    or negated (real x, an odd quantity such as a phase angle).

    x:         [N, 1, h, w] complex or real field in the convention of theta
    theta:     [N, 1, h, w] orientation of each source pixel
    theta_ref: [N, 1, H, W] reference orientation of each target pixel
    size:      target (H, W)
    Same sample positions and border handling as isu.upsample (grid_sample bicubic, align_corners=True, border).
    """
    h, w = x.shape[-2:]
    H, W = int(size[0]), int(size[1])
    real_dtype = theta.dtype
    fy = 2.0 ** round(torch.log2(torch.tensor(H / h)).item())
    fx = 2.0 ** round(torch.log2(torch.tensor(W / w)).item())
    py = torch.arange(H, device=x.device, dtype=real_dtype) / fy
    px = torch.arange(W, device=x.device, dtype=real_dtype) / fx
    by, bx = torch.floor(py), torch.floor(px)
    wy, wx = _cubic_weights(py - by), _cubic_weights(px - bx)
    by, bx = by.long(), bx.long()

    out = torch.zeros(x.shape[:2] + (H, W), dtype=x.dtype, device=x.device)
    for j in range(4):
        iy = (by - 1 + j).clamp(0, h - 1)
        for i in range(4):
            ix = (bx - 1 + i).clamp(0, w - 1)
            tap = x[..., iy[:, None], ix[None, :]]
            sg = align_sign(theta[..., iy[:, None], ix[None, :]], theta_ref)
            tap = torch.complex(tap.real, sg * tap.imag) if torch.is_complex(tap) else sg * tap
            out = out + (wy[j][:, None] * wx[i][None, :]) * tap
    return out


def get_phi_quick_aligned(c, f, dir_c, dir_f):
    """get_phi_quick with c upsampled in the orientation convention of f (dir_f); phi is in that convention too."""
    c_up = upsample_aligned(c, dir_c, dir_f, f.shape[2:])
    c_hat = c_up**2 / (c_up.abs() + 1e-6)
    return c_hat * torch.conj(f)


def extrapolate_phi_aligned(phi, f, dir_f, dir_out):
    """
    extrapolate_phi(phi, f, steps_up=1) followed by the upsampling to the size of dir_out (Worm.repair_phase), with
    phi and f upsampled in the orientation convention of dir_out (both are in the convention of dir_f).
    The result is at the size of dir_out and in its convention.
    """
    size_up = dir_out.shape[2:]
    # upscale_complex_phase: |angle| and magnitude do not depend on the convention, only the sign of the angle does
    phi_angle = phi.angle()
    phi_angle_abs_up = isu.upsample(phi_angle.abs(), size_up, mode='bicubic')
    phi_angle_up = phi_angle_abs_up * torch.sign(upsample_aligned(phi_angle, dir_f, dir_out, size_up))
    phi_up = torch.polar(isu.upsample(phi.abs(), size_up, mode='bicubic'), phi_angle_up)

    f_up = upsample_aligned(f, dir_f, dir_out, size_up)
    f_phase = 2 * f_up.angle() - phi_up.angle()
    f_phase = (f_phase + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.polar(f_up.abs(), f_phase)
