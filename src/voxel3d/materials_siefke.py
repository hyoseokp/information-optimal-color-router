"""Siefke ALD TiO2 (refractiveindex.info, tabulated n,k) as material "TiO2_siefke" for the cr_itd_v2 forward.

install() patches cr_itd_v2.forward.materials (and the names torcwa_local / local_response bound at import) so both
the paper's reference forward and voxel3d see the same permittivity; every other material falls through unchanged.
"""
from __future__ import annotations

import numpy as np

YML = str(__import__("pathlib").Path(__file__).with_name("Siefke_TiO2.yml"))
NAME = "TiO2_siefke"
_table = None


def _load(path=YML):
    global _table
    if _table is None:
        rows, on = [], False
        for line in open(path, encoding="utf-8"):
            if line.strip().startswith("data: |"):
                on = True
                continue
            if on:
                t = line.split()
                if len(t) != 3:
                    break
                rows.append([float(v) for v in t])
        _table = np.array(rows)
    return _table


def nk(wavelength_um: float) -> complex:
    t = _load()
    return complex(np.interp(wavelength_um, t[:, 0], t[:, 1]), np.interp(wavelength_um, t[:, 0], t[:, 2]))


def install() -> None:
    import cr_itd_v2.forward.materials as m
    import cr_itd_v2.forward.torcwa_local as tl
    import cr_itd_v2.forward.local_response as lr
    if getattr(m, "_siefke_installed", False):
        return
    ri0, pm0 = m.refractive_index, m.permittivity

    def refractive_index(material, wavelength_um=None):
        if material == NAME:
            return nk(wavelength_um)
        return ri0(material, wavelength_um)

    def permittivity(material, wavelength_um=None):
        if material == NAME:
            return nk(wavelength_um) ** 2
        return pm0(material, wavelength_um)

    m.refractive_index, m.permittivity = refractive_index, permittivity
    tl.permittivity = permittivity
    lr.refractive_index = refractive_index
    m._siefke_installed = True
