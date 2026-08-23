"""Reference demonstration on synthetic inputs; not the paper's numbers.

Runs on CPU, needs no GPU, no RCWA solver, and no external data.  It builds a
finite-NA pupil quadrature, then evaluates the closed-form imaging-information
objective on a synthetic linear-Gaussian channel.
"""

import math

import torch

from cr_itd_v2.forward import PupilSpec, pupil_quadrature, relative_illumination
from cr_itd_v2.information import joint_target_information


def main() -> None:
    torch.manual_seed(0)

    # 1. Finite-NA, field-dependent incoherent pupil ensemble.
    spec = PupilSpec(
        numerical_aperture=0.30,
        chief_ray_angle_deg=2.5,
        chief_ray_azimuth_deg=22.5,
        radial_order=2,
        azimuth_count=8,
    )
    rays = pupil_quadrature(spec)
    weight_sum = sum(ray.weight for ray in rays)
    print(f"pupil rays                 : {len(rays)}")
    print(
        "max ray angle (deg)        : "
        f"{max(math.degrees(ray.theta_reference_rad) for ray in rays):.4f}"
    )
    print(f"pupil weight sum           : {weight_sum:.12f}")
    print(
        "relative illumination      : "
        f"{relative_illumination(spec.chief_ray_angle_deg):.12f}"
    )

    # 2. Synthetic linear-Gaussian channel.
    #    K spectral latent components, M measured wells, D target channels.
    latent_channels = 8
    measurements = 4
    target_channels = 3
    dtype = torch.float64

    factor = torch.randn(latent_channels, latent_channels, dtype=dtype)
    scene_covariance = factor @ factor.T + latent_channels * torch.eye(
        latent_channels, dtype=dtype
    )

    # Deterministic linear target (for example a colour-matching transform)
    # plus a small independent target-nuisance floor, which keeps the joint
    # scene/target covariance valid.
    target_transform = torch.randn(target_channels, latent_channels, dtype=dtype)
    scene_target_cross = scene_covariance @ target_transform.T
    target_covariance = (
        target_transform @ scene_covariance @ target_transform.T
        + 1.0e-2 * torch.eye(target_channels, dtype=dtype)
    )

    # Measurement matrix A (non-negative, as a spectral response would be) and
    # diagonal measurement noise.
    measurement_matrix = torch.rand(measurements, latent_channels, dtype=dtype)
    noise_covariance = torch.diag(
        torch.full((measurements,), 0.5, dtype=dtype)
    )

    # 3. Imaging information I_img = 1/2 log2(det Sigma_Z / det Sigma_{Z|Y}).
    result = joint_target_information(
        measurement_matrix,
        scene_covariance,
        scene_target_cross,
        target_covariance,
        noise_covariance,
    )
    bits = float(result.information_bits)
    print(f"imaging information (bits) : {bits:.6f}")
    print(f"finite                     : {math.isfinite(bits)}")


if __name__ == "__main__":
    main()
