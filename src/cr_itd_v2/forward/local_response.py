"""Differentiable flat-field camera-pupil response for four detector wells."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
import torch.utils.checkpoint

from cr_itd_v2.forward.contracts import (
    ACTIVE_WELL_ORDER,
    PupilRay,
    PupilSpec,
    RESPONSE_AXIS_ORDER,
    RESPONSE_UNITS,
    StackSpec,
)
from cr_itd_v2.forward.materials import refractive_index
from cr_itd_v2.forward.pupil import pupil_quadrature
from cr_itd_v2.forward.torcwa_local import (
    ENERGY_GUARD_EPS,
    ENERGY_GUARD_RETRY_NUDGES,
    detector_axes,
    power_honest_well_powers,
    record_energy_guard_event,
    solve_plane_wave,
)


SOURCE_LOCAL_RUNNER_SHA256 = (
    "898b8d33136a7175b61e5d41a3c30a1ae066e3b2b304d08eb838b3190aa0d259"
)
SOURCE_LOCAL_BASE_SHA256 = (
    "b8bbe2b311acf0cc47efa6b361d8146cf7445d71e2b3d3b6065e3deff5af0e7e"
)
POLARIZATION_ORDER = ("x", "y")


@dataclass(frozen=True, slots=True)
class LocalSpectralResponse:
    """Transmission-rescaled local allocation proxy with explicit metadata."""

    tabs: torch.Tensor
    wavelengths_nm: tuple[float, ...]
    pupil_weight_sum: float
    ray_count: int
    well_order: tuple[str, ...] = ACTIVE_WELL_ORDER
    axis_order: tuple[str, ...] = RESPONSE_AXIS_ORDER
    units: str = RESPONSE_UNITS
    polarization_order: tuple[str, ...] = POLARIZATION_ORDER

    def __post_init__(self) -> None:
        if self.tabs.ndim != 2 or tuple(self.tabs.shape) != (
            len(ACTIVE_WELL_ORDER),
            len(self.wavelengths_nm),
        ):
            raise ValueError("tabs must have shape [4, wavelength]")
        if self.well_order != ACTIVE_WELL_ORDER:
            raise ValueError("local response well-order drift")
        if self.axis_order != RESPONSE_AXIS_ORDER or self.units != RESPONSE_UNITS:
            raise ValueError("local response metadata drift")
        if self.polarization_order != POLARIZATION_ORDER:
            raise ValueError("local response must contain full x/y polarization")
        if self.ray_count <= 0 or self.pupil_weight_sum <= 0:
            raise ValueError("local response requires positive ray count/weight")

    @property
    def response(self) -> torch.Tensor:
        return self.tabs


def _wavelength_tuple(values: Sequence[float] | torch.Tensor) -> tuple[float, ...]:
    if torch.is_tensor(values):
        if values.ndim != 1:
            raise ValueError("wavelengths_nm must be one-dimensional")
        raw = values.detach().cpu().tolist()
    else:
        raw = list(values)
    wavelengths = tuple(float(item) for item in raw)
    if not wavelengths:
        raise ValueError("wavelengths_nm must be non-empty")
    if any(not math.isfinite(item) or not 400.0 <= item <= 700.0 for item in wavelengths):
        raise ValueError("wavelengths_nm must be finite and lie in [400, 700]")
    if any(right <= left for left, right in zip(wavelengths, wavelengths[1:])):
        raise ValueError("wavelengths_nm must be strictly increasing")
    return wavelengths


def _refracted_incidence_theta(
    ray: PupilRay,
    reference_index: float,
    wavelength_um: float,
    stack: StackSpec,
) -> float:
    superstrate_index = refractive_index(
        stack.superstrate_material, wavelength_um
    )
    superstrate_real = (
        float(superstrate_index.real)
        if isinstance(superstrate_index, complex)
        else float(superstrate_index)
    )
    if not math.isfinite(superstrate_real) or superstrate_real <= 0:
        raise ValueError("superstrate refractive index must be finite and positive")
    sine_inside = (
        float(reference_index)
        * math.sin(ray.theta_reference_rad)
        / superstrate_real
    )
    if not 0 <= sine_inside < 1:
        raise ValueError("pupil ray cannot refract into the stack superstrate")
    return math.asin(sine_inside)


def _guarded_well_powers(
    density: torch.Tensor,
    wavelength_um: float,
    incidence_theta: float,
    phi_rad: float,
    polarization: str,
    stack: StackSpec,
    axis_x: torch.Tensor,
    axis_y: torch.Tensor,
) -> torch.Tensor:
    """One solved polarization, re-solved at alternate nudges if singular.

    The check reads a detached scalar off the already-computed well powers, so
    a healthy solve returns exactly the tensors it always did -- bit-identical
    values and an unchanged autograd graph.  See the guard note in
    ``torcwa_local`` for why the retry ladder exists and why no constant
    ``rayleigh_frequency_nudge`` can be globally safe.
    """

    solution = solve_plane_wave(
        density, wavelength_um, incidence_theta, phi_rad, polarization, stack
    )
    well_power = power_honest_well_powers(solution, axis_x, axis_y, stack)
    transmission = float(well_power.detach().sum()) / stack.cell_area_um2
    if transmission <= 1.0 + ENERGY_GUARD_EPS:
        return well_power
    for nudge in ENERGY_GUARD_RETRY_NUDGES:
        retry_stack = dataclasses.replace(stack, rayleigh_frequency_nudge=nudge)
        solution = solve_plane_wave(
            density, wavelength_um, incidence_theta, phi_rad, polarization,
            retry_stack,
        )
        retry_power = power_honest_well_powers(
            solution, axis_x, axis_y, retry_stack
        )
        retry_transmission = (
            float(retry_power.detach().sum()) / retry_stack.cell_area_um2
        )
        if retry_transmission <= 1.0 + ENERGY_GUARD_EPS:
            record_energy_guard_event(
                {
                    "wavelength_um": wavelength_um,
                    "ray": f"theta={incidence_theta:.5f},phi={phi_rad:.5f}",
                    "polarization": polarization,
                    "transmission": transmission,
                    "nudge_used": nudge,
                    "transmission_retry": retry_transmission,
                }
            )
            return retry_power
    raise RuntimeError(
        f"energy guard: transmission {transmission:.4f} at wavelength "
        f"{wavelength_um:.4f} um, theta {incidence_theta:.5f}, phi "
        f"{phi_rad:.5f}, pol {polarization} exceeds {1.0 + ENERGY_GUARD_EPS} "
        f"at every retry nudge {ENERGY_GUARD_RETRY_NUDGES}; refusing to "
        f"return an unchecked number"
    )


def _one_wavelength_response(
    density: torch.Tensor,
    wavelength_nm: float,
    stack: StackSpec,
    pupil_spec: PupilSpec,
    rays: tuple[PupilRay, ...],
) -> torch.Tensor:
    wavelength_um = wavelength_nm / 1000.0
    axis_x, axis_y = detector_axes(stack, density.device)
    accumulated: torch.Tensor | None = None
    for ray in rays:
        incidence_theta = _refracted_incidence_theta(
            ray, pupil_spec.reference_index, wavelength_um, stack
        )
        polarization_sum: torch.Tensor | None = None
        for polarization in POLARIZATION_ORDER:
            well_power = _guarded_well_powers(
                density,
                wavelength_um,
                incidence_theta,
                ray.phi_rad,
                polarization,
                stack,
                axis_x,
                axis_y,
            )
            polarization_sum = (
                well_power
                if polarization_sum is None
                else polarization_sum + well_power
            )
        ray_average = polarization_sum * (ray.weight / len(POLARIZATION_ORDER))
        accumulated = (
            ray_average if accumulated is None else accumulated + ray_average
        )
    if accumulated is None:  # guarded by the public entry point
        raise RuntimeError("empty pupil reached wavelength response")
    return accumulated / stack.cell_area_um2


def evaluate_local_spectral_response(
    density: torch.Tensor,
    wavelengths_nm: Sequence[float] | torch.Tensor,
    *,
    stack: StackSpec,
    pupil_spec: PupilSpec,
    rays: Iterable[PupilRay] | None = None,
    checkpoint_wavelengths: bool = False,
) -> LocalSpectralResponse:
    """Evaluate the four-well local allocation proxy under an incoherent pupil.

    ``rays`` may be an exact-weight subset for memory-light pupil-block VJPs.
    The returned response is linear in those ray weights.  No illuminant, QE,
    exposure, clipping, or post-hoc DC rescaling is applied here.  The four
    entries sum to exact S-matrix transmission, but their split is inherited
    from detector-plane quadrant ``|E|^2`` rather than local Poynting flux,
    silicon absorption, or carrier collection.
    """

    wavelength_values = _wavelength_tuple(wavelengths_nm)
    ray_values = tuple(pupil_quadrature(pupil_spec) if rays is None else rays)
    if not ray_values or any(not isinstance(ray, PupilRay) for ray in ray_values):
        raise ValueError("rays must be a non-empty iterable of PupilRay values")
    rows = []
    for wavelength_nm in wavelength_values:
        function = lambda value, wavelength=wavelength_nm: _one_wavelength_response(
            value, wavelength, stack, pupil_spec, ray_values
        )
        response = (
            torch.utils.checkpoint.checkpoint(
                function, density, use_reentrant=False
            )
            if checkpoint_wavelengths
            else function(density)
        )
        rows.append(response)
    tabs = torch.stack(rows, dim=1).to(torch.float64)
    if not bool(torch.isfinite(tabs.detach()).all()):
        raise RuntimeError("local spectral response contains non-finite values")
    if bool((tabs.detach() < 0).any()):
        raise RuntimeError("local spectral response contains negative power")
    return LocalSpectralResponse(
        tabs=tabs,
        wavelengths_nm=wavelength_values,
        pupil_weight_sum=float(sum(ray.weight for ray in ray_values)),
        ray_count=len(ray_values),
    )


def local_camera_pupil_response(
    density: torch.Tensor,
    wavelengths_nm: Sequence[float] | torch.Tensor,
    *,
    stack: StackSpec,
    pupil_spec: PupilSpec,
    rays: Iterable[PupilRay] | None = None,
    checkpoint_wavelengths: bool = False,
) -> torch.Tensor:
    """Return only ``Tabs[well,wavelength]`` for optimizer composition."""

    return evaluate_local_spectral_response(
        density,
        wavelengths_nm,
        stack=stack,
        pupil_spec=pupil_spec,
        rays=rays,
        checkpoint_wavelengths=checkpoint_wavelengths,
    ).tabs


__all__ = [
    "LocalSpectralResponse",
    "POLARIZATION_ORDER",
    "SOURCE_LOCAL_BASE_SHA256",
    "SOURCE_LOCAL_RUNNER_SHA256",
    "evaluate_local_spectral_response",
    "local_camera_pupil_response",
]
