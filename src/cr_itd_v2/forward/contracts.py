"""Immutable contracts for the active local colour-router forward model.

The local model deliberately stops at a four-well, flat-field spectral
response.  Spatial kernels and sensor-grid sampling are downstream.  Every
length is in micrometres, density arrays are indexed ``[x, y]``, and active
measurement rows use ``R, G2, G1, B``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


ACTIVE_WELL_ORDER = ("R", "G2", "G1", "B")
DENSITY_AXIS_ORDER = ("x", "y")
RESPONSE_AXIS_ORDER = ("well", "wavelength")
RESPONSE_UNITS = (
    "dimensionless exact-total-transmission-rescaled detector-plane "
    "|E|^2 quadrant allocation proxy"
)

LEGACY_SOURCE_SHA256 = {
    "cell_model.py": "fd6b7297df8f1ffe7f587025e5b27e39bd9b451aeb5f92d923e509ea00d27165",
    "focused_na.py": "9c9c6a4a7ce5e117232590dc5616722aed724348e51bb7c4f210a585f2b2a6f9",
    "illumination.py": "b138e9ab8601377027cdfa6ca7534dd33fd0ef50c65e1bcd3e9bd4cb7e01f4c1",
    "materials.py": "ded90d4bb53100a7c4b8b297a6d8c7025aa8d9aa5e87bca9f38f87eb945c5492",
    "materials_nk.json": "83ab10195ccd5a5332516158fcc8bbf880bd7772d6f016804a392a73b5e3adf4",
}


def _positive_pair(value: tuple[float, float], name: str) -> None:
    if len(value) != 2 or any(not math.isfinite(float(x)) or float(x) <= 0 for x in value):
        raise ValueError(f"{name} must contain two finite positive values")


def _positive_int_pair(value: tuple[int, int], name: str) -> None:
    if (
        len(value) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in value)
    ):
        raise ValueError(f"{name} must contain two positive integers")


@dataclass(frozen=True, slots=True)
class StackSpec:
    """Periodic stack of one or more patterned layers, and the detector plane.

    ``layer_heights_um`` empty selects the single-layer geometry, and
    ``design_height_um`` is that layer's thickness.  Giving it N thicknesses stacks N independently patterned layers separated by
    ``layer_gap_um`` of unpatterned spacer, in which case the density carries a
    leading axis of length N and ``design_height_um`` is unused.
    """

    period_um: tuple[float, float] = (2.0, 2.0)
    design_height_um: float = 0.75
    detector_z_um: float = 2.77
    superstrate_material: str = "SiO2_nk"
    design_material: str = "SiN_nk"
    substrate_material: str = "SiO2_nk"
    density_shape: tuple[int, int] = (128, 128)
    fourier_orders: tuple[int, int] = (8, 8)
    detector_samples: tuple[int, int] = (24, 24)
    rayleigh_frequency_nudge: float = 1.0e-4
    layer_heights_um: tuple[float, ...] = ()
    layer_gap_um: float = 0.0
    layer_gap_material: str = ""
    # What fills the unpatterned part of a design layer.  Empty means the
    # superstrate, which is right for a single exposed layer.  A stack is
    # different: the layers are planarised in
    # oxide and only the top surface meets air, so the fill and the superstrate
    # are not the same material and cannot share one field.
    design_fill_material: str = ""
    # The sensor mosaic repeat.  Zero selects the default convention -- the four
    # wells are the four quadrants of the period, which is only right when the
    # period IS the pixel pitch.  A positive value declares the pixel pitch
    # explicitly, so a router period of an integer multiple of it (a supercell)
    # scores each well row as the union of its like-role 1x1 sites across the
    # covered pixels: the average pixel's response, in the same fraction units.
    mosaic_pitch_um: float = 0.0

    def __post_init__(self) -> None:
        _positive_pair(self.period_um, "period_um")
        _positive_int_pair(self.density_shape, "density_shape")
        _positive_int_pair(self.detector_samples, "detector_samples")
        if (
            len(self.fourier_orders) != 2
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 0
                for x in self.fourier_orders
            )
        ):
            raise ValueError("fourier_orders must contain two non-negative integers")
        for name in ("design_height_um", "detector_z_um"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.rayleigh_frequency_nudge) or not (
            0.0 <= self.rayleigh_frequency_nudge < 1.0e-2
        ):
            raise ValueError("rayleigh_frequency_nudge must lie in [0, 1e-2)")
        for name in (
            "superstrate_material",
            "design_material",
            "substrate_material",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if any(
            not math.isfinite(float(h)) or float(h) <= 0 for h in self.layer_heights_um
        ):
            raise ValueError("layer_heights_um entries must be finite and positive")
        if not math.isfinite(self.layer_gap_um) or self.layer_gap_um < 0:
            raise ValueError("layer_gap_um must be finite and non-negative")
        # The detector sits at an absolute depth, so a stack that grows past it
        # would be scored behind its own last layer rather than in front of it.
        # Only multi-layer stacks are checked, so a single-layer geometry may
        # still place the detector plane wherever the caller intends.
        if self.layer_heights_um and self.detector_z_um <= self.total_height_um:
            raise ValueError(
                f"detector_z_um {self.detector_z_um} must exceed the stack's "
                f"total height {self.total_height_um}"
            )
        if self.mosaic_pitch_um != 0.0:
            pitch = float(self.mosaic_pitch_um)
            if not math.isfinite(pitch) or pitch <= 0:
                raise ValueError("mosaic_pitch_um must be finite and positive")
            for axis_period in self.period_um:
                multiple = float(axis_period) / pitch
                if abs(multiple - round(multiple)) > 1.0e-9 or round(multiple) < 1:
                    raise ValueError(
                        "period_um must be a positive integer multiple of "
                        "mosaic_pitch_um in both axes"
                    )

    @property
    def cell_area_um2(self) -> float:
        return float(self.period_um[0] * self.period_um[1])

    @property
    def patterned_heights_um(self) -> tuple[float, ...]:
        """Thickness of each patterned layer, top first."""

        return (
            tuple(float(h) for h in self.layer_heights_um)
            if self.layer_heights_um
            else (float(self.design_height_um),)
        )

    @property
    def patterned_layer_count(self) -> int:
        return len(self.patterned_heights_um)

    @property
    def total_height_um(self) -> float:
        heights = self.patterned_heights_um
        return float(sum(heights) + self.layer_gap_um * (len(heights) - 1))

    @property
    def gap_material(self) -> str:
        return self.layer_gap_material or self.fill_material

    @property
    def fill_material(self) -> str:
        return self.design_fill_material or self.superstrate_material


@dataclass(frozen=True, slots=True)
class PupilSpec:
    """Field-dependent shifted-disk pupil contract in the air reference frame."""

    numerical_aperture: float = 0.30
    chief_ray_angle_deg: float = 2.5
    chief_ray_azimuth_deg: float = 22.5
    radial_order: int = 2
    azimuth_count: int = 8
    reference_index: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_index) or self.reference_index <= 0:
            raise ValueError("reference_index must be finite and positive")
        if not math.isfinite(self.numerical_aperture) or not (
            0 < self.numerical_aperture < self.reference_index
        ):
            raise ValueError("numerical_aperture must lie in (0, reference_index)")
        if not math.isfinite(self.chief_ray_angle_deg) or not (
            0 <= self.chief_ray_angle_deg < 90
        ):
            raise ValueError("chief_ray_angle_deg must lie in [0, 90)")
        if not math.isfinite(self.chief_ray_azimuth_deg):
            raise ValueError("chief_ray_azimuth_deg must be finite")
        if (
            isinstance(self.radial_order, bool)
            or not isinstance(self.radial_order, int)
            or self.radial_order <= 0
        ):
            raise ValueError("radial_order must be a positive integer")
        if (
            isinstance(self.azimuth_count, bool)
            or not isinstance(self.azimuth_count, int)
            or self.azimuth_count <= 0
        ):
            raise ValueError("azimuth_count must be a positive integer")

    @property
    def ray_count(self) -> int:
        return int(self.radial_order * self.azimuth_count)


@dataclass(frozen=True, slots=True)
class PupilRay:
    """One incoherent pupil direction, defined before entrance refraction."""

    theta_reference_rad: float
    phi_rad: float
    sx_reference: float
    sy_reference: float
    weight: float

    def __post_init__(self) -> None:
        values = (
            self.theta_reference_rad,
            self.phi_rad,
            self.sx_reference,
            self.sy_reference,
            self.weight,
        )
        if any(not math.isfinite(float(x)) for x in values):
            raise ValueError("pupil-ray values must be finite")
        if not 0 <= self.theta_reference_rad < math.pi / 2:
            raise ValueError("theta_reference_rad must lie in [0, pi/2)")
        if self.weight <= 0:
            raise ValueError("pupil-ray weight must be positive")
        transverse = math.hypot(self.sx_reference, self.sy_reference)
        if not math.isclose(
            transverse,
            math.sin(self.theta_reference_rad),
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ValueError("pupil-ray direction cosines disagree with theta")


DEFAULT_R1A_STACK = StackSpec()
DEFAULT_R1A_PUPIL = PupilSpec()


__all__ = [
    "ACTIVE_WELL_ORDER",
    "DEFAULT_R1A_PUPIL",
    "DEFAULT_R1A_STACK",
    "DENSITY_AXIS_ORDER",
    "LEGACY_SOURCE_SHA256",
    "PupilRay",
    "PupilSpec",
    "RESPONSE_AXIS_ORDER",
    "RESPONSE_UNITS",
    "StackSpec",
]
