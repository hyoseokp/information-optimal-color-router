"""Pinned dispersive materials for the local TORCWA forward model.

The tabulated values are embedded so no external table has to be resolved at
run time.  They are copied from the source whose SHA-256 is recorded in
:mod:`cr_itd_v2.forward.contracts`.
"""

from __future__ import annotations

from bisect import bisect_right
import math

from cr_itd_v2.forward.contracts import LEGACY_SOURCE_SHA256


SOURCE_MATERIALS_SHA256 = LEGACY_SOURCE_SHA256["materials.py"]
SOURCE_NK_TABLE_SHA256 = LEGACY_SOURCE_SHA256["materials_nk.json"]

_WAVELENGTHS_NM = tuple(float(x) for x in range(400, 701, 10))
_N_TABLES = {
    "SiN_nk": (
        2.1004, 2.0953, 2.0906, 2.0863, 2.0823, 2.0786, 2.0751, 2.0719,
        2.0688, 2.0660, 2.0634, 2.0609, 2.0585, 2.0563, 2.0543, 2.0523,
        2.0504, 2.0487, 2.0470, 2.0454, 2.0439, 2.0425, 2.0411, 2.0398,
        2.0386, 2.0374, 2.0362, 2.0351, 2.0341, 2.0331, 2.0321,
    ),
    "SiO2_nk": (
        1.4701, 1.4691, 1.4681, 1.4672, 1.4663, 1.4656, 1.4648, 1.4641,
        1.4635, 1.4629, 1.4623, 1.4618, 1.4613, 1.4608, 1.4603, 1.4599,
        1.4595, 1.4591, 1.4587, 1.4584, 1.4580, 1.4577, 1.4574, 1.4571,
        1.4568, 1.4565, 1.4563, 1.4560, 1.4558, 1.4555, 1.4553,
    ),
    # Exploratory, not part of the pinned set above.  The SiN and SiO2 tables are
    # copied from the source whose SHA-256 is recorded in contracts; this one is a
    # Cauchy fit, n = 2.2317 + 0.05368/lam^2 + 0.002129/lam^4 with lam in um,
    # through representative amorphous/ALD TiO2 anchors (2.65 at 400 nm falling to
    # 2.35 at 700 nm), reproducing them to 0.003.  Absorption is set to zero: the
    # band gap sits near 360 nm, so k is negligible across 400-700 nm for the
    # amorphous films this stands in for.  Any result quoted from it has to say
    # that the index came from a literature-typical fit rather than a measured
    # film, because the phase budget scales directly with n and this is the one
    # number the comparison against SiN turns on.
    "TiO2_nk": (
        2.6504, 2.6264, 2.6044, 2.5843, 2.5658, 2.5487, 2.5329, 2.5183,
        2.5048, 2.4922, 2.4805, 2.4696, 2.4593, 2.4498, 2.4408, 2.4324,
        2.4245, 2.4171, 2.4101, 2.4035, 2.3972, 2.3913, 2.3857, 2.3805,
        2.3754, 2.3707, 2.3661, 2.3618, 2.3577, 2.3538, 2.3501,
    ),
}
_K_TABLES = {
    "SiN_nk": (0.0,) * len(_WAVELENGTHS_NM),
    "SiO2_nk": (0.0,) * len(_WAVELENGTHS_NM),
    "TiO2_nk": (0.0,) * len(_WAVELENGTHS_NM),
}
_CONSTANT_INDEX = {
    "air": 1.0,
    "SiN": 2.0,
    "SiO2": 1.46,
}
SUPPORTED_MATERIALS = tuple(sorted((*_CONSTANT_INDEX, *_N_TABLES)))


def _wavelength_nm(wavelength_um: float) -> float:
    if isinstance(wavelength_um, bool):
        raise ValueError("wavelength_um must be a finite positive scalar")
    value = float(wavelength_um)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("wavelength_um must be a finite positive scalar")
    return value * 1000.0


def _linear_table_value(values: tuple[float, ...], wavelength_nm: float) -> float:
    if wavelength_nm <= _WAVELENGTHS_NM[0]:
        return float(values[0])
    if wavelength_nm >= _WAVELENGTHS_NM[-1]:
        return float(values[-1])
    upper = bisect_right(_WAVELENGTHS_NM, wavelength_nm)
    lower = upper - 1
    x0 = _WAVELENGTHS_NM[lower]
    x1 = _WAVELENGTHS_NM[upper]
    fraction = (wavelength_nm - x0) / (x1 - x0)
    return float(values[lower] + fraction * (values[upper] - values[lower]))


def refractive_index(material: str, wavelength_um: float | None = None) -> complex | float:
    """Return ``n + i k`` using the pinned 400--700 nm linear table."""

    if material in _CONSTANT_INDEX:
        return float(_CONSTANT_INDEX[material])
    if material not in _N_TABLES:
        raise ValueError(
            f"unknown material {material!r}; supported={SUPPORTED_MATERIALS}"
        )
    if wavelength_um is None:
        raise ValueError(f"material {material!r} requires wavelength_um")
    wavelength_nm = _wavelength_nm(wavelength_um)
    n_value = _linear_table_value(_N_TABLES[material], wavelength_nm)
    k_value = _linear_table_value(_K_TABLES[material], wavelength_nm)
    return complex(n_value, k_value)


def permittivity(material: str, wavelength_um: float | None = None) -> complex | float:
    """Return relative permittivity ``(n + i k)**2``."""

    index = refractive_index(material, wavelength_um)
    return index * index


def wavelength_table_nm() -> tuple[float, ...]:
    """Return the immutable source wavelength grid."""

    return _WAVELENGTHS_NM


__all__ = [
    "SOURCE_MATERIALS_SHA256",
    "SOURCE_NK_TABLE_SHA256",
    "SUPPORTED_MATERIALS",
    "permittivity",
    "refractive_index",
    "wavelength_table_nm",
]
