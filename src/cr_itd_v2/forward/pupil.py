"""Field-dependent incoherent shifted-disk pupil quadrature."""

from __future__ import annotations

import math

import numpy as np

from cr_itd_v2.forward.contracts import (
    LEGACY_SOURCE_SHA256,
    PupilRay,
    PupilSpec,
)


SOURCE_ILLUMINATION_SHA256 = LEGACY_SOURCE_SHA256["illumination.py"]


def relative_illumination(chief_ray_angle_deg: float) -> float:
    """Return the legacy local-model relative illumination ``cos(CRA)^4``."""

    angle = float(chief_ray_angle_deg)
    if not math.isfinite(angle) or not 0 <= angle < 90:
        raise ValueError("chief_ray_angle_deg must lie in [0, 90)")
    return math.cos(math.radians(angle)) ** 4


def pupil_quadrature(spec: PupilSpec) -> tuple[PupilRay, ...]:
    """Build the immutable Gauss--Legendre radial/uniform-azimuth pupil.

    Directions are expressed in the air/reference frame.  Refraction into the
    explicit stack superstrate happens inside the TORCWA response calculation.
    Weights are incoherent intensity weights and sum to ``cos(CRA)^4``.
    """

    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(
        spec.radial_order
    )
    radial_nodes = 0.5 * (radial_nodes + 1.0)
    radial_weights = 0.5 * radial_weights
    pupil_radius = spec.numerical_aperture / spec.reference_index
    chief_sine = math.sin(math.radians(spec.chief_ray_angle_deg))
    chief_phi = math.radians(spec.chief_ray_azimuth_deg)
    chief_x = chief_sine * math.cos(chief_phi)
    chief_y = chief_sine * math.sin(chief_phi)

    raw: list[tuple[float, float, float, float, float]] = []
    for radial_node, radial_weight in zip(radial_nodes, radial_weights):
        radius = pupil_radius * math.sqrt(float(radial_node))
        area_weight = (
            math.pi
            * pupil_radius**2
            * float(radial_weight)
            / spec.azimuth_count
        )
        for azimuth_index in range(spec.azimuth_count):
            pupil_phi = (
                math.pi / 4.0
                + 2.0 * math.pi * azimuth_index / spec.azimuth_count
            )
            sx = chief_x + radius * math.cos(pupil_phi)
            sy = chief_y + radius * math.sin(pupil_phi)
            transverse = math.hypot(sx, sy)
            if transverse >= 1.0:
                raise ValueError(
                    "shifted pupil contains an evanescent reference-frame sample: "
                    f"|s|={transverse:.6g}"
                )
            theta = math.asin(transverse)
            phi = math.atan2(sy, sx)
            raw.append((theta, phi, sx, sy, area_weight / math.cos(theta)))

    target_sum = relative_illumination(spec.chief_ray_angle_deg)
    raw_sum = sum(item[-1] for item in raw)
    if raw_sum <= 0 or not math.isfinite(raw_sum):
        raise RuntimeError("pupil quadrature has invalid raw weight sum")
    rays = tuple(
        PupilRay(
            theta_reference_rad=theta,
            phi_rad=phi,
            sx_reference=sx,
            sy_reference=sy,
            weight=weight * target_sum / raw_sum,
        )
        for theta, phi, sx, sy, weight in raw
    )
    if len(rays) != spec.ray_count:
        raise RuntimeError("pupil quadrature ray-count contract failed")
    if not math.isclose(
        sum(ray.weight for ray in rays),
        target_sum,
        rel_tol=0.0,
        abs_tol=2.0e-15,
    ):
        raise RuntimeError("pupil quadrature normalization failed")
    return rays


__all__ = [
    "SOURCE_ILLUMINATION_SHA256",
    "pupil_quadrature",
    "relative_illumination",
]
