"""Density-independent constants for every (wavelength, ray, nudge) solve.

Everything here depends only on wavelength, incidence direction and the
Rayleigh frequency nudge -- never on the design -- so it is built once and
reused for every optimizer step. The formulas are torcwa 0.1.4.2's
(``rcwa._kvectors``, ``source_fourier``, ``S_parameters`` 'ps' branch,
``field_xy`` output-layer branch), written in structured form:

* A homogeneous-medium matrix V (and so every input/output S-matrix block) is
  2x2 blocks of diagonal N x N matrices.  torcwa stores them dense and pays
  full matmuls and a dense inverse; here they are four length-N vectors
  ("BDD": block-diagonal-diagonal) and their inverse is an analytic 2x2
  inverse per diffraction order.
* The transmitted power is linear in the output field, so the p/s projection,
  evanescent masking and power normalisation collapse into constant
  coefficient vectors applied to S11 @ E_i.
* The detector-plane synthesis exp(i w (kx x + ky y)) factorises because kx
  depends only on the x order and ky only on the y order, so the 24x24 field
  is two small matmuls rather than a 576 x 289 contraction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

TORCWA_PI = 3.141592652589793  # torcwa.rcwa's own constant (sic), kept for bit-level fidelity


@dataclass
class BDD:
    """2x2 block matrix whose blocks are diagonal: [[a, b], [c, d]], each [..., N]."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor

    def __add__(self, o: "BDD") -> "BDD":
        return BDD(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    def __sub__(self, o: "BDD") -> "BDD":
        return BDD(self.a - o.a, self.b - o.b, self.c - o.c, self.d - o.d)

    def scale(self, s) -> "BDD":
        return BDD(self.a * s, self.b * s, self.c * s, self.d * s)

    def matmul_bdd(self, o: "BDD") -> "BDD":
        return BDD(self.a * o.a + self.b * o.c, self.a * o.b + self.b * o.d,
                   self.c * o.a + self.d * o.c, self.c * o.b + self.d * o.d)

    def inverse(self) -> "BDD":
        det = self.a * self.d - self.b * self.c
        return BDD(self.d / det, -self.b / det, -self.c / det, self.a / det)

    def left(self, x: torch.Tensor) -> torch.Tensor:
        """self @ x for x [..., 2N, k]: a row operation, no matmul."""
        n = self.a.shape[-1]
        x1, x2 = x[..., :n, :], x[..., n:, :]
        return torch.cat((self.a[..., :, None] * x1 + self.b[..., :, None] * x2,
                          self.c[..., :, None] * x1 + self.d[..., :, None] * x2), dim=-2)

    def right(self, x: torch.Tensor) -> torch.Tensor:
        """x @ self for x [..., k, 2N]: a column operation, no matmul."""
        n = self.a.shape[-1]
        x1, x2 = x[..., :, :n], x[..., :, n:]
        return torch.cat((x1 * self.a[..., None, :] + x2 * self.c[..., None, :],
                          x1 * self.b[..., None, :] + x2 * self.d[..., None, :]), dim=-1)


def _kz(eps, kx, ky):
    kz = torch.sqrt(eps - kx ** 2 - ky ** 2)
    return torch.where(torch.imag(kz) < 0, torch.conj(kz), kz)


def homogeneous_v(eps, kx, ky) -> BDD:
    """torcwa's E-to-H matrix of a homogeneous medium (mu = 1), in BDD form."""
    kz = _kz(eps, kx, ky)
    return BDD(-ky * kx / kz, -kz - ky ** 2 / kz, kz + kx ** 2 / kz, kx * ky / kz)


@dataclass
class SolveGeometry:
    """Batched constants for B solves.  All tensors carry a leading B axis."""

    omega: torch.Tensor          # [B]      complex64, 2*pi*freq
    kx: torch.Tensor             # [B, N]   complex64
    ky: torch.Tensor             # [B, N]
    eps_out: torch.Tensor        # [B]
    lam_index: torch.Tensor      # [B]      int64, which wavelength (for the shared E^-1)
    ray_index: torch.Tensor      # [B]
    nudge: torch.Tensor          # [B]      float64
    vf_inv: BDD                  # free-space V^-1
    sin11: BDD
    sin12: BDD
    sout11: BDD
    sout21: BDD
    e_in: torch.Tensor           # [B, 2N, 2]  source columns for x and y polarisation
    out_phase: torch.Tensor      # [B, 2N]     exp(i w kz_out z_det) for the output-layer field
    vo: BDD                      # output-medium V (for H in the output layer)
    coef: torch.Tensor           # [B, 4, N]   p/s projection x power norm, evanescent zeroed
    phase_x: torch.Tensor        # [B, Sx, Ox] exp(i w kx_order x)
    phase_y: torch.Tensor        # [B, Sy, Oy]
    thickness: float
    orders: tuple[int, int]


def build_geometry(solves, *, stack, orders, detector_samples, period, device,
                   eps_in_fn, eps_out_fn, cdt=torch.complex64) -> SolveGeometry:
    """Constants for a list of solves ``(lam_idx, ray_idx, wl_um, theta, phi, nudge)``."""
    rdt = torch.float64 if cdt == torch.complex128 else torch.float32
    ox, oy = orders
    order_x = torch.arange(-ox, ox + 1, device=device, dtype=torch.int64)
    order_y = torch.arange(-oy, oy + 1, device=device, dtype=torch.int64)
    n_orders = (2 * ox + 1) * (2 * oy + 1)
    lx, ly = period
    ref = n_orders // 2  # the (0, 0) order in torcwa's x-major/y-minor layout

    rows = {k: [] for k in ("omega", "kx", "ky", "eps_out", "eps_in", "kx0", "ky0", "freq")}
    lam_idx, ray_idx, nudges, phis, thetas = [], [], [], [], []
    for (li, ri, wl, theta, phi, nudge) in solves:
        freq = (1.0 / wl) * (1.0 + nudge)
        freq_t = torch.as_tensor(freq, dtype=cdt, device=device)
        eps_in = torch.as_tensor(eps_in_fn(wl), dtype=cdt, device=device)
        eps_out = torch.as_tensor(eps_out_fn(wl), dtype=cdt, device=device)
        inc = torch.as_tensor(theta, dtype=cdt, device=device)
        azi = torch.as_tensor(phi, dtype=cdt, device=device)
        n_in = torch.real(torch.sqrt(eps_in))
        kx0 = n_in * torch.sin(inc) * torch.cos(azi)
        ky0 = n_in * torch.sin(inc) * torch.sin(azi)
        gx, gy = 1 / (lx * freq_t), 1 / (ly * freq_t)
        kx_ord = kx0 + order_x * gx
        ky_ord = ky0 + order_y * gy
        kxg, kyg = torch.meshgrid(kx_ord, ky_ord, indexing="ij")
        rows["omega"].append(torch.as_tensor(2 * TORCWA_PI * freq, dtype=cdt, device=device))
        rows["kx"].append(kxg.reshape(-1))
        rows["ky"].append(kyg.reshape(-1))
        rows["eps_out"].append(eps_out)
        rows["eps_in"].append(eps_in)
        rows["kx0"].append(kx_ord)
        rows["ky0"].append(ky_ord)
        rows["freq"].append(freq_t)
        lam_idx.append(li)
        ray_idx.append(ri)
        nudges.append(nudge)
        phis.append(phi)
        thetas.append(theta)

    omega = torch.stack(rows["omega"])
    kx = torch.stack(rows["kx"])
    ky = torch.stack(rows["ky"])
    eps_out = torch.stack(rows["eps_out"])
    eps_in = torch.stack(rows["eps_in"])
    kx_axis = torch.stack(rows["kx0"])          # [B, Ox]
    ky_axis = torch.stack(rows["ky0"])          # [B, Oy]

    one = torch.ones_like(eps_in)
    vf = homogeneous_v(one[:, None], kx, ky)
    vi = homogeneous_v(eps_in[:, None], kx, ky)
    vo = homogeneous_v(eps_out[:, None], kx, ky)

    # input layer S-matrix (torcwa _kvectors, 'Sin')
    t1 = (vf + vi).inverse()
    t2 = vf - vi
    sin11 = t1.matmul_bdd(vi).scale(2)
    sin12 = t1.matmul_bdd(t2)                   # Rb S12 = +Vtmp1 Vtmp2
    # output layer S-matrix (torcwa _kvectors, 'Sout')
    t1o = (vf + vo).inverse()
    t2o = vf - vo
    sout11 = t1o.matmul_bdd(vf).scale(2)
    sout21 = t1o.matmul_bdd(t2o)                # Rf S21 = +Vtmp1 Vtmp2

    # source vectors (torcwa source_fourier, notation 'ps', forward, order (0,0))
    kt = torch.sqrt(kx ** 2 + ky ** 2)
    kz_in_real = torch.abs(torch.real(torch.sqrt(eps_in[:, None] - kx ** 2 - ky ** 2)))
    inc_in = torch.atan2(torch.real(kt), kz_in_real)
    azi_all = torch.atan2(torch.real(ky), torch.real(kx))
    B = kx.shape[0]
    e_in = torch.zeros((B, 2 * n_orders, 2), dtype=cdt, device=device)
    for b in range(B):
        phi = phis[b]
        for p, (ap, as_) in enumerate(((math.cos(phi), -math.sin(phi)), (math.sin(phi), math.cos(phi)))):
            ci = torch.cos(inc_in[b, ref])
            e_in[b, ref, p] = ap * ci * torch.cos(azi_all[b, ref]) + as_ * (-torch.sin(azi_all[b, ref]))
            e_in[b, ref + n_orders, p] = ap * ci * torch.sin(azi_all[b, ref]) + as_ * torch.cos(azi_all[b, ref])

    # output-layer field phase (torcwa field_xy, layer_num == layer_N, forward)
    kz_out_field = _kz(eps_out[:, None], kx, ky)
    zph = torch.exp(1j * omega[:, None] * kz_out_field * float(stack.detector_z_um))
    out_phase = torch.cat((zph, zph), dim=-1)

    # transmitted-power coefficients (torcwa S_parameters 'ps', forward transmission, power_norm)
    evanescent = 1e-3
    kz_out_c = torch.sqrt(eps_out[:, None] - kx ** 2 - ky ** 2)
    order_evan = torch.abs(torch.real(kz_out_c) / torch.imag(kz_out_c)) < evanescent
    order_kz = torch.abs(torch.real(kz_out_c))
    order_inc = torch.atan2(torch.real(kt), order_kz)
    order_azi = azi_all
    kz_in_c = torch.sqrt(eps_in[:, None] - kx ** 2 - ky ** 2)
    in_evan = torch.abs(torch.real(kz_in_c) / torch.imag(kz_in_c)) < evanescent
    kz_in_norm = torch.where(in_evan, torch.zeros_like(kz_in_c.real), torch.real(kz_in_c))
    kz_out_norm = torch.where(order_evan, torch.abs(torch.real(kz_out_c)), torch.real(kz_out_c))
    norm = torch.sqrt(kz_out_norm / kz_in_norm[:, ref:ref + 1])
    cos_i = torch.cos(order_inc)
    cp_x = torch.cos(order_azi) / cos_i
    cp_y = torch.sin(order_azi) / cos_i
    cs_x = -torch.sin(order_azi)
    cs_y = torch.cos(order_azi)
    coef = torch.stack((cp_x, cp_y, cs_x, cs_y), dim=1) * norm[:, None, :]
    coef = torch.where(order_evan[:, None, :], torch.zeros_like(coef), coef)
    coef = torch.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0).to(cdt)

    # separable detector synthesis
    sx, sy = detector_samples
    ax = (torch.arange(sx, device=device, dtype=rdt) + 0.5) * (lx / sx)
    ay = (torch.arange(sy, device=device, dtype=rdt) + 0.5) * (ly / sy)
    phase_x = torch.exp(1j * omega[:, None, None] * kx_axis[:, None, :] * ax[None, :, None])
    phase_y = torch.exp(1j * omega[:, None, None] * ky_axis[:, None, :] * ay[None, :, None])

    return SolveGeometry(
        omega=omega, kx=kx, ky=ky, eps_out=eps_out,
        lam_index=torch.as_tensor(lam_idx, device=device),
        ray_index=torch.as_tensor(ray_idx, device=device),
        nudge=torch.as_tensor(nudges, dtype=torch.float64),
        vf_inv=vf.inverse(), sin11=sin11, sin12=sin12, sout11=sout11, sout21=sout21,
        e_in=e_in, out_phase=out_phase, vo=vo, coef=coef,
        phase_x=phase_x, phase_y=phase_y,
        thickness=float(stack.design_height_um), orders=orders,
    )
