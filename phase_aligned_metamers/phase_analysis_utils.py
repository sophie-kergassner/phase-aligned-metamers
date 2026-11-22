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
