"""CPU parity: voxel3d multi-layer forward vs the paper's torcwa reference forward (torcwa, per solve).

Small stack (3 touching TiO2/air layers, 16x16 density, orders (2,2), 2 wavelengths, 2 rays), complex128 for
the voxel3d side.  Gates: Tabs relative error, gradient cosine vs the reference autograd, central finite
differences of the voxel3d c128 objective, slot non-coupling, 1-layer == crfast.

usage: python voxel3d/test_voxel_parity.py [--layers 3]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cr_itd_v2.forward import DEFAULT_R1A_PUPIL, DEFAULT_R1A_STACK  # noqa: E402
from cr_itd_v2.forward.local_response import local_camera_pupil_response  # noqa: E402
from cr_itd_v2.forward.pupil import pupil_quadrature  # noqa: E402
from voxel3d import materials_siefke  # noqa: E402
from voxel3d.forward import FastPupilForward  # noqa: E402

materials_siefke.install()

TOL_TABS_REL = 5e-4
TOL_GRAD_COS = 0.999
TOL_FD_REL = 2e-3


def build(n_layers, h=0.1):
    stack = dataclasses.replace(
        DEFAULT_R1A_STACK, superstrate_material="SiO2_nk", substrate_material="air", design_material=materials_siefke.NAME,
        design_fill_material="air", density_shape=(16, 16), fourier_orders=(2, 2), detector_samples=(8, 8),
        design_height_um=h, layer_heights_um=(h,) * n_layers if n_layers > 1 else (),
        detector_z_um=n_layers * h + 0.45)
    pupil = dataclasses.replace(DEFAULT_R1A_PUPIL, radial_order=1, azimuth_count=2)
    return stack, pupil


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()
    wl = (450.0, 600.0)
    nl = args.layers
    stack, pupil = build(nl)
    rays = pupil_quadrature(pupil)
    gen = torch.Generator().manual_seed(3)
    rho0 = (0.1 + 0.8 * torch.rand((nl, 16, 16), generator=gen)).to(torch.float64)
    wt = torch.rand((4, len(wl)), generator=gen, dtype=torch.float64) + 0.5
    out = {}

    fwd = FastPupilForward(stack=stack, pupil_spec=pupil, wavelengths_nm=wl, device="cpu", dtype=torch.complex128, chunk=3)
    stack_ref = stack

    def ref_obj(rho):
        tabs = local_camera_pupil_response(rho.to(torch.float32), wl, stack=stack_ref, pupil_spec=pupil, rays=rays)
        return (tabs.to(torch.float64) * wt).sum(), tabs

    def vox_obj(rho, f=fwd):
        tabs = f.responses(rho[None])[0]
        return (tabs * wt).sum(), tabs

    r = rho0.clone().requires_grad_(True)
    v_ref, tabs_ref = ref_obj(r)
    v_ref.backward()
    g_ref = r.grad.detach().clone()
    r = rho0.clone().requires_grad_(True)
    v_vox, tabs_vox = vox_obj(r)
    v_vox.backward()
    g_vox = r.grad.detach().clone()

    rel = float((tabs_vox.detach() - tabs_ref.detach()).abs().max() / tabs_ref.detach().abs().max())
    cos = float(torch.nn.functional.cosine_similarity(g_vox.flatten(), g_ref.flatten(), dim=0))
    out.update(layers=nl, tabs_rel_err=rel, grad_cos_vs_reference=cos, value_vox=float(v_vox), value_ref=float(v_ref))

    eps = 1e-5
    idx = [(0, 3, 5), (nl - 1, 8, 8), (nl // 2, 12, 2), (nl - 1, 0, 15)]
    fd_rel = []
    for (l, i, j) in idx:
        up, dn = rho0.clone(), rho0.clone()
        up[l, i, j] += eps
        dn[l, i, j] -= eps
        with torch.no_grad():
            fd = (vox_obj(up)[0] - vox_obj(dn)[0]) / (2 * eps)
        fd_rel.append(float(abs(g_vox[l, i, j] - fd) / max(abs(float(fd)), 1e-12)))
    out["fd_rel_err"] = fd_rel

    r2 = torch.stack([rho0, (1 - rho0)]).clone().requires_grad_(True)
    t2 = fwd.responses(r2, tags=("a", "b"))
    ((t2[0] * wt).sum() + (t2[1] * wt).sum()).backward()
    out["slot_coupling_err"] = float((t2[0].detach() - tabs_vox.detach()).abs().max())
    out["slot_grad_err"] = float((r2.grad[0] - g_vox).abs().max() / g_vox.abs().max())

    ok = (rel < TOL_TABS_REL and cos > TOL_GRAD_COS and max(fd_rel) < TOL_FD_REL
          and out["slot_coupling_err"] < 1e-9 and out["slot_grad_err"] < 1e-8)
    out["pass"] = bool(ok)
    print(json.dumps(out, indent=1))
    Path(__file__).with_name(f"parity_{nl}layer.json").write_text(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
