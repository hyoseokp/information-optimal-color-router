"""crfast vs the reference forward (cr_itd_v2 + torcwa), CPU, complex128, 2 wavelengths x 2 rays."""
import dataclasses

import numpy as np
import torch

from cr_itd_v2.forward import DEFAULT_R1A_PUPIL, DEFAULT_R1A_STACK
from cr_itd_v2.forward.local_response import local_camera_pupil_response
from cr_itd_v2.forward.pupil import pupil_quadrature
from crfast.forward import FastPupilForward

WLS = (450.0, 540.0)


def _setup():
    stack = dataclasses.replace(DEFAULT_R1A_STACK, superstrate_material="air",
                                design_height_um=0.600, detector_z_um=1.250)
    pupil = dataclasses.replace(DEFAULT_R1A_PUPIL, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5,
                                radial_order=2, azimuth_count=8)
    rays = tuple(pupil_quadrature(pupil))[:2]
    return stack, pupil, rays


def _density(stack):
    g = torch.Generator().manual_seed(1)
    return (0.5 + 0.3 * (2 * torch.rand(tuple(stack.density_shape), generator=g) - 1)).float()


def test_response_and_gradient_match_reference():
    stack, pupil, rays = _setup()
    rho0 = _density(stack)
    w = torch.randn(4, len(WLS), generator=torch.Generator().manual_seed(2), dtype=torch.float64)

    r = rho0.clone().requires_grad_(True)
    ref = local_camera_pupil_response(r, WLS, stack=stack, pupil_spec=pupil, rays=rays)
    (ref * w).sum().backward()
    g_ref = r.grad.clone()

    fwd = FastPupilForward(stack=stack, pupil_spec=pupil, wavelengths_nm=WLS, device=torch.device("cpu"),
                           rays=rays, dtype=torch.complex128, verbose_guard=False)
    r2 = rho0.clone().requires_grad_(True)
    out = fwd.response(r2)
    (out * w).sum().backward()

    rel = float((out.detach() - ref.detach()).abs().max() / ref.detach().abs().max())
    cos = float(torch.nn.functional.cosine_similarity(r2.grad.flatten().float(), g_ref.flatten(), dim=0))
    print(f"response rel {rel:.2e}  gradient cosine {cos:.6f}")
    assert rel < 2e-4 and cos > 0.999
