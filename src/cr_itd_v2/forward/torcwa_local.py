"""Differentiable single-ray TORCWA primitives for the local CR model."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import importlib
import math
from typing import Any, Sequence

import torch

from cr_itd_v2.forward.contracts import (
    ACTIVE_WELL_ORDER,
    LEGACY_SOURCE_SHA256,
    StackSpec,
)
from cr_itd_v2.forward.materials import permittivity


SOURCE_CELL_MODEL_SHA256 = LEGACY_SOURCE_SHA256["cell_model.py"]
SOURCE_FOCUSED_NA_SHA256 = LEGACY_SOURCE_SHA256["focused_na.py"]

# Energy-conservation guard.  The frequency nudge that steers a solve away
# from a Rayleigh anomaly can itself place some (pattern, ray, wavelength)
# combination on a numerically singular complex64 solve, which returns an
# S-matrix transmission far above unity.  No single constant nudge is
# globally safe, so the violation is detected per solve and only that solve
# is retried at the alternative nudges below; a healthy solve is untouched
# and stays bit-identical.  If every nudge trips, the caller raises rather
# than returning an unchecked number.
ENERGY_GUARD_EPS = 0.02
ENERGY_GUARD_RETRY_NUDGES = (0.0, 5.0e-4)
_ENERGY_GUARD_EVENTS: list[dict] = []


def record_energy_guard_event(event: dict) -> None:
    """Append one retry record and print it so run logs show contamination."""

    _ENERGY_GUARD_EVENTS.append(dict(event))
    print(
        "ENERGY-GUARD retry: wavelength {wavelength_um:.4f} um  ray {ray} "
        "pol {polarization}  T={transmission:.4f} -> nudge {nudge_used:g} "
        "T={transmission_retry:.4f}".format(**event),
        flush=True,
    )


def pop_energy_guard_events() -> list[dict]:
    """Return and clear the retry records accumulated since the last call."""

    events = list(_ENERGY_GUARD_EVENTS)
    _ENERGY_GUARD_EVENTS.clear()
    return events


@dataclass(frozen=True, slots=True)
class SolvedPlaneWave:
    """One solved Cartesian-polarized plane wave and its source convention."""

    simulation: Any
    wavelength_um: float
    incidence_theta_rad: float
    azimuth_rad: float
    polarization: str


def _torcwa_module() -> Any:
    try:
        return importlib.import_module("torcwa")
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "TORCWA is required for the local forward model; install the "
            "optional 'rcwa' extra"
        ) from exc


def _validate_density(density: torch.Tensor, stack: StackSpec) -> torch.Tensor:
    """Check one patterned layer's density; see ``_layer_densities`` for stacks."""

    value = torch.as_tensor(density)
    if value.dtype != torch.float32 or value.ndim != 2:
        raise ValueError("density must be a two-dimensional float32 tensor")
    if tuple(map(int, value.shape)) != stack.density_shape:
        raise ValueError(
            f"density shape {tuple(value.shape)} does not match {stack.density_shape}"
        )
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError("density contains non-finite values")
    if bool((value.detach() < 0).any()) or bool((value.detach() > 1).any()):
        raise ValueError("density must lie in [0, 1]")
    return value


def _layer_densities(density: torch.Tensor, stack: StackSpec) -> tuple[torch.Tensor, ...]:
    """Split a design into one density per patterned layer, top layer first.

    A single-layer stack still accepts the plain two-dimensional density every
    caller wrote before layers existed, and also accepts a leading axis of one,
    so the two spellings cannot disagree about which layer is which.
    """

    value = torch.as_tensor(density)
    count = stack.patterned_layer_count
    if value.ndim == 2:
        if count != 1:
            raise ValueError(
                f"stack has {count} patterned layers; density must have a leading axis"
            )
        return (_validate_density(value, stack),)
    if value.ndim != 3:
        raise ValueError("density must have two or three dimensions")
    if int(value.shape[0]) != count:
        raise ValueError(
            f"density carries {int(value.shape[0])} layers, stack expects {count}"
        )
    return tuple(_validate_density(value[index], stack) for index in range(count))


def _validate_design(density: torch.Tensor, stack: StackSpec) -> torch.Tensor:
    """Validate a whole design -- every layer -- and return it unchanged."""

    _layer_densities(density, stack)
    return torch.as_tensor(density)


def build_permittivity(
    density: torch.Tensor, wavelength_um: float, stack: StackSpec
) -> torch.Tensor:
    """Map ``density[x,y]`` to the dispersive design-layer permittivity."""

    value = _validate_density(density, stack)
    background = torch.as_tensor(
        permittivity(stack.fill_material, wavelength_um),
        dtype=torch.complex64,
        device=value.device,
    )
    design = torch.as_tensor(
        permittivity(stack.design_material, wavelength_um),
        dtype=torch.complex64,
        device=value.device,
    )
    return background + value.to(torch.complex64) * (design - background)


def _add_patterned_layers(
    simulation, density: torch.Tensor, wavelength_um: float, stack: StackSpec
) -> None:
    """Add every patterned layer, with unpatterned spacers in between.

    Layers are added in propagation order, so index 0 is the one the light meets
    first.  The spacer is a homogeneous layer of ``gap_material``; with a zero
    gap the patterned layers touch and none is inserted.
    """

    layers = _layer_densities(density, stack)
    heights = stack.patterned_heights_um
    gap = float(stack.layer_gap_um)
    gap_eps = permittivity(stack.gap_material, wavelength_um)
    for index, (layer, height) in enumerate(zip(layers, heights)):
        if index and gap > 0.0:
            simulation.add_layer(thickness=gap, eps=gap_eps)
        simulation.add_layer(
            thickness=height,
            eps=build_permittivity(layer, wavelength_um, stack),
        )


def all_diffraction_orders(orders: tuple[int, int]) -> list[list[int]]:
    """Return TORCWA order pairs in explicit x-major/y-minor order."""

    order_x, order_y = orders
    return [
        [index_x, index_y]
        for index_x in range(-order_x, order_x + 1)
        for index_y in range(-order_y, order_y + 1)
    ]


def source_jones_ps(azimuth_rad: float, polarization: str) -> tuple[float, float]:
    """Project a fixed Cartesian x/y source onto the ray's p/s basis."""

    phi = float(azimuth_rad)
    if polarization == "x":
        return math.cos(phi), -math.sin(phi)
    if polarization == "y":
        return math.sin(phi), math.cos(phi)
    raise ValueError("polarization must be 'x' or 'y'")


def solve_plane_wave(
    density: torch.Tensor,
    wavelength_um: float,
    incidence_theta_rad: float,
    azimuth_rad: float,
    polarization: str,
    stack: StackSpec,
) -> SolvedPlaneWave:
    """Solve one oblique plane wave with the explicit embedded stack."""

    value = _validate_design(density, stack)
    wavelength = float(wavelength_um)
    theta = float(incidence_theta_rad)
    phi = float(azimuth_rad)
    if not math.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength_um must be finite and positive")
    if not math.isfinite(theta) or not 0 <= theta < math.pi / 2:
        raise ValueError("incidence_theta_rad must lie in [0, pi/2)")
    if not math.isfinite(phi):
        raise ValueError("azimuth_rad must be finite")
    amplitude_p, amplitude_s = source_jones_ps(phi, polarization)
    torcwa = _torcwa_module()
    frequency = (1.0 / wavelength) * (1.0 + stack.rayleigh_frequency_nudge)
    simulation = torcwa.rcwa(
        freq=frequency,
        order=list(stack.fourier_orders),
        L=list(stack.period_um),
        dtype=torch.complex64,
        device=value.device,
    )
    simulation.add_input_layer(
        eps=permittivity(stack.superstrate_material, wavelength)
    )
    simulation.add_output_layer(
        eps=permittivity(stack.substrate_material, wavelength)
    )
    simulation.set_incident_angle(inc_ang=theta, azi_ang=phi)
    _add_patterned_layers(simulation, value, wavelength, stack)
    simulation.solve_global_smatrix()
    simulation.source_planewave(
        amplitude=[[amplitude_p, amplitude_s]],
        direction="forward",
        notation="ps",
    )
    return SolvedPlaneWave(
        simulation=simulation,
        wavelength_um=wavelength,
        incidence_theta_rad=theta,
        azimuth_rad=phi,
        polarization=polarization,
    )


def well_powers_over_polarizations(
    density: torch.Tensor,
    wavelength_um: float,
    incidence_theta_rad: float,
    azimuth_rad: float,
    stack: StackSpec,
    axis_x: torch.Tensor,
    axis_y: torch.Tensor,
    polarizations: Sequence[str],
) -> tuple[torch.Tensor, ...]:
    """Well powers for several source polarizations off one solved structure.

    The layer eigenproblem and the global S-matrix depend on the geometry, the
    wavelength and the incidence direction.  Polarization enters only through
    the source vector, so :func:`solve_plane_wave` called once per polarization
    repeats every expensive step -- for the two-polarization average this pupil
    uses, exactly half the work.  Building once and re-sourcing reproduces the
    per-polarization forward values to 1e-12.

    The source is set immediately before each field evaluation because
    ``field_xy`` reads whatever source the simulation currently holds.

    NOT FOR DIFFERENTIABLE USE, and the guard below enforces it.  Re-sourcing
    mutates simulation state that the first polarization's graph still
    references, so the returned gradient is wrong.  Use it only for
    no-gradient work -- response caches, hard-mask checkpoints, evaluation --
    where it is exact and halves the cost.
    """

    value = _validate_design(density, stack)
    wavelength = float(wavelength_um)
    theta = float(incidence_theta_rad)
    phi = float(azimuth_rad)
    if not math.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength_um must be finite and positive")
    if not math.isfinite(theta) or not 0 <= theta < math.pi / 2:
        raise ValueError("incidence_theta_rad must lie in [0, pi/2)")
    if not math.isfinite(phi):
        raise ValueError("azimuth_rad must be finite")
    if not polarizations:
        raise ValueError("polarizations must be non-empty")
    if torch.is_grad_enabled() and value.requires_grad:
        raise RuntimeError(
            "well_powers_over_polarizations is not differentiable; the re-sourced "
            "simulation returns a corrupted gradient. Wrap the call in torch.no_grad() "
            "or use solve_plane_wave per polarization."
        )

    torcwa = _torcwa_module()

    def build(nudge: float):
        frequency = (1.0 / wavelength) * (1.0 + nudge)
        simulation = torcwa.rcwa(
            freq=frequency,
            order=list(stack.fourier_orders),
            L=list(stack.period_um),
            dtype=torch.complex64,
            device=value.device,
        )
        simulation.add_input_layer(
            eps=permittivity(stack.superstrate_material, wavelength)
        )
        simulation.add_output_layer(
            eps=permittivity(stack.substrate_material, wavelength)
        )
        simulation.set_incident_angle(inc_ang=theta, azi_ang=phi)
        _add_patterned_layers(simulation, value, wavelength, stack)
        simulation.solve_global_smatrix()
        return simulation

    def solved_power(simulation, polarization: str) -> torch.Tensor:
        amplitude_p, amplitude_s = source_jones_ps(phi, polarization)
        simulation.source_planewave(
            amplitude=[[amplitude_p, amplitude_s]],
            direction="forward",
            notation="ps",
        )
        solution = SolvedPlaneWave(
            simulation=simulation,
            wavelength_um=wavelength,
            incidence_theta_rad=theta,
            azimuth_rad=phi,
            polarization=polarization,
        )
        return power_honest_well_powers(solution, axis_x, axis_y, stack)

    simulation = build(stack.rayleigh_frequency_nudge)
    retry_simulations: dict[float, Any] = {}
    results: list[torch.Tensor] = []
    for polarization in polarizations:
        power = solved_power(simulation, polarization)
        transmission = float(power.detach().sum()) / stack.cell_area_um2
        if transmission > 1.0 + ENERGY_GUARD_EPS:
            # Singular solve; see the guard note above ENERGY_GUARD_EPS.
            for nudge in ENERGY_GUARD_RETRY_NUDGES:
                if nudge not in retry_simulations:
                    retry_simulations[nudge] = build(nudge)
                retry_power = solved_power(retry_simulations[nudge], polarization)
                retry_transmission = (
                    float(retry_power.detach().sum()) / stack.cell_area_um2
                )
                if retry_transmission <= 1.0 + ENERGY_GUARD_EPS:
                    record_energy_guard_event(
                        {
                            "wavelength_um": wavelength,
                            "ray": f"theta={theta:.5f},phi={phi:.5f}",
                            "polarization": polarization,
                            "transmission": transmission,
                            "nudge_used": nudge,
                            "transmission_retry": retry_transmission,
                        }
                    )
                    power = retry_power
                    break
            else:
                raise RuntimeError(
                    f"energy guard: transmission {transmission:.4f} at "
                    f"wavelength {wavelength:.4f} um, theta {theta:.5f}, "
                    f"phi {phi:.5f}, pol {polarization} exceeds "
                    f"{1.0 + ENERGY_GUARD_EPS} at every retry nudge "
                    f"{ENERGY_GUARD_RETRY_NUDGES}; refusing to return an "
                    f"unchecked number"
                )
        results.append(power)
    return tuple(results)


def detector_axes(
    stack: StackSpec, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return endpoint-free midpoint axes for periodic detector integration."""

    count_x, count_y = stack.detector_samples
    axis_x = (
        torch.arange(count_x, device=device, dtype=torch.float32) + 0.5
    ) * (stack.period_um[0] / count_x)
    axis_y = (
        torch.arange(count_y, device=device, dtype=torch.float32) + 0.5
    ) * (stack.period_um[1] / count_y)
    return axis_x, axis_y


def _quadrant_masks(
    axis_x: torch.Tensor, axis_y: torch.Tensor, stack: StackSpec
) -> tuple[torch.Tensor, ...]:
    if stack.mosaic_pitch_um != 0.0:
        # Supercell: the mosaic repeats at the declared pixel pitch, so each
        # well row is the union of its like-role sites across every covered
        # pixel.  With period == pitch this reduces exactly to the quadrant
        # split below, which is kept as its own branch so that case is
        # evaluated by identical arithmetic in either spelling.
        pitch = float(stack.mosaic_pitch_um)
        half = pitch / 2.0
        mod_x = axis_x - pitch * torch.floor(axis_x / pitch)
        mod_y = axis_y - pitch * torch.floor(axis_y / pitch)
        low_x = (mod_x < half).reshape(-1, 1)
        high_x = ~low_x
        low_y = (mod_y < half).reshape(1, -1)
        high_y = ~low_y
        masks = {
            "R": low_x & low_y,
            "G2": high_x & low_y,
            "G1": low_x & high_y,
            "B": high_x & high_y,
        }
        return tuple(masks[name] for name in ACTIVE_WELL_ORDER)
    split_x = stack.period_um[0] / 2.0
    split_y = stack.period_um[1] / 2.0
    low_x = (axis_x < split_x).reshape(-1, 1)
    high_x = ~low_x
    low_y = (axis_y < split_y).reshape(1, -1)
    high_y = ~low_y
    masks = {
        "R": low_x & low_y,
        "G2": high_x & low_y,
        "G1": low_x & high_y,
        "B": high_x & high_y,
    }
    return tuple(masks[name] for name in ACTIVE_WELL_ORDER)


def quadrant_field_powers(
    electric_field: Sequence[torch.Tensor],
    axis_x: torch.Tensor,
    axis_y: torch.Tensor,
    stack: StackSpec,
) -> torch.Tensor:
    """Integrate the inherited detector-plane ``|E|^2`` allocation proxy.

    These quadrant integrals determine only relative well allocation.  They
    are not local z-directed Poynting flux, silicon absorption, or charge
    collection probabilities.
    """

    if len(electric_field) != 3:
        raise ValueError("electric_field must contain Ex, Ey, Ez")
    expected = (axis_x.numel(), axis_y.numel())
    if any(tuple(component.shape) != expected for component in electric_field):
        raise ValueError(f"electric-field components must have shape {expected}")
    intensity = sum(torch.abs(component).square() for component in electric_field)
    area_element = (
        stack.period_um[0]
        / axis_x.numel()
        * stack.period_um[1]
        / axis_y.numel()
    )
    return torch.stack(
        [
            (intensity * mask).sum() * area_element
            for mask in _quadrant_masks(axis_x, axis_y, stack)
        ]
    )


def transmitted_power(solution: SolvedPlaneWave, stack: StackSpec) -> torch.Tensor:
    """Return exact power-normalized total transmission for the source Jones vector."""

    simulation = solution.simulation
    orders = all_diffraction_orders(stack.fourier_orders)
    columns = {
        label: simulation.S_parameters(
            orders=orders,
            direction="forward",
            port="transmission",
            polarization=label,
            power_norm=True,
        )
        for label in ("pp", "ps", "sp", "ss")
    }
    amplitude_p, amplitude_s = source_jones_ps(
        solution.azimuth_rad, solution.polarization
    )
    output_p = amplitude_p * columns["pp"] + amplitude_s * columns["ps"]
    output_s = amplitude_p * columns["sp"] + amplitude_s * columns["ss"]
    return (torch.abs(output_p).square() + torch.abs(output_s).square()).sum()


def power_honest_well_powers(
    solution: SolvedPlaneWave,
    axis_x: torch.Tensor,
    axis_y: torch.Tensor,
    stack: StackSpec,
) -> torch.Tensor:
    """Return exact total transmission split by the detector-plane proxy.

    The four entries sum to the power-normalized S-matrix transmission.  Their
    relative split inherits the quadrant ``|E|^2`` proxy documented in
    :func:`quadrant_field_powers`; calling the entries collected electrons
    without the later detector model would be incorrect.
    """

    simulation = solution.simulation
    electric_field, _ = simulation.field_xy(
        simulation.layer_N,
        axis_x,
        axis_y,
        z_prop=float(stack.detector_z_um),
    )
    field_powers = quadrant_field_powers(
        electric_field, axis_x, axis_y, stack
    )
    field_total = field_powers.sum()
    if not bool(torch.isfinite(field_total.detach())) or not bool(
        (field_total.detach() > 0).item()
    ):
        raise RuntimeError("detector-plane field power is not finite and positive")
    exact_transmission = transmitted_power(solution, stack)
    if not bool(torch.isfinite(exact_transmission.detach())) or bool(
        (exact_transmission.detach() < 0).item()
    ):
        raise RuntimeError("TORCWA transmission is not finite and non-negative")
    fractions = field_powers / field_total
    return fractions * exact_transmission * stack.cell_area_um2


__all__ = [
    "ENERGY_GUARD_EPS",
    "ENERGY_GUARD_RETRY_NUDGES",
    "SOURCE_CELL_MODEL_SHA256",
    "SOURCE_FOCUSED_NA_SHA256",
    "SolvedPlaneWave",
    "pop_energy_guard_events",
    "record_energy_guard_event",
    "all_diffraction_orders",
    "build_permittivity",
    "detector_axes",
    "power_honest_well_powers",
    "quadrant_field_powers",
    "solve_plane_wave",
    "source_jones_ps",
    "transmitted_power",
    "well_powers_over_polarizations",
]
