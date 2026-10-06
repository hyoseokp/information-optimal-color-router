"""voxel3d: crfast extended to a stack of independently patterned layers (no spacer).

Every (density slot, layer) pair is one eigenproblem row; the S-matrix chain over layers is
the Redheffer star product with one 2N solve per layer.  A single layer reduces to crfast.

---- crfast header ----
Batched, structured RCWA pupil response -- the paper's forward, one launch.

Reproduces cr_itd_v2.forward.local_response.local_camera_pupil_response
(torcwa 0.1.4.2 per solve, energy guard, incoherent x/y + pupil average) for
the single-patterned-layer stack, but computes all (wavelength, ray) solves as
one batch and both polarisations off one structure solve.  Differences from
the reference are algebraic rewrites of the same quantities, never physics
changes; tests/test_parity.py holds them to the reference numerically.

Rewrites, each exact in exact arithmetic:
  * both polarisations share the eigenproblem and S-matrix (two source
    columns); the reference re-solves per polarisation because torcwa mutates
    state, which a functional batch does not;
  * E^-1 is per wavelength, shared by every ray;
  * P and Q are assembled by row/column scaling of E^-1 and E, not by matmuls
    with dense diagonal Kx/Ky;
  * the 4N x 4N mode-coupling inverse is [[A, C], [C, A]]^-1 via (A + C)^-1 and
    (A - C)^-1 (two 2N inverses instead of one 4N inverse, ~4x fewer flops);
  * the single-layer S-matrix is symmetric (S12 = S21, S22 = S11);
  * input/output S-matrices are block-diagonal-diagonal and applied as row or
    column operations;
  * only S11 @ E_i is needed downstream, so the global S-matrix is never
    formed: two linear solves replace torcwa's four inverses per product.
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from . import eig as eigmod
from .eig import eig_attach
from .geometry import SolveGeometry, build_geometry

ENERGY_GUARD_EPS = 0.02
ENERGY_GUARD_RETRY_NUDGES = (0.0, 5.0e-4)
WELL_ORDER = ("R", "G2", "G1", "B")


class _ChunkedGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, pairs, nudges, gsegs, e_all, einv_all):
        wls, trs, store = [], [], []
        for s, e in gsegs:
            g, rows = owner._chunk_geo(pairs, nudges, s, e)
            vals, x = owner._chunk_eig(e_all, einv_all, rows)
            store.append((vals.cpu(), x.cpu()))
            wl, tr = owner._chunk_wells(e_all, einv_all, g, rows, vals, x, e - s, False)
            wls.append(wl)
            trs.append(tr)
            del vals, x
        ctx.owner, ctx.pairs, ctx.nudges, ctx.gsegs, ctx.store = owner, pairs, nudges, gsegs, store
        ctx.save_for_backward(e_all, einv_all)
        wl, tr = torch.cat(wls), torch.cat(trs)
        ctx.mark_non_differentiable(tr)
        return wl, tr

    @staticmethod
    def backward(ctx, gw, gt):
        e_all, einv_all = ctx.saved_tensors
        owner = ctx.owner
        e_l = e_all.detach().requires_grad_(True)
        i_l = einv_all.detach().requires_grad_(True)
        o = 0
        for (s, e), (vals, x) in zip(ctx.gsegs, ctx.store):
            g, rows = owner._chunk_geo(ctx.pairs, ctx.nudges, s, e)
            dev = e_all.device
            with torch.enable_grad():
                wl, _ = owner._chunk_wells(e_l, i_l, g, rows, vals.to(dev), x.to(dev), e - s, True)
                torch.autograd.backward([wl], [gw[o:o + e - s]])
            o += e - s
            del wl
        ctx.store = None
        return None, None, None, None, e_l.grad, i_l.grad


class FastPupilForward:
    def __init__(self, *, stack, pupil_spec, wavelengths_nm: Sequence[float], device,
                 rays=None, chunk: int | None = None, verbose_guard: bool = True,
                 dtype: torch.dtype = torch.complex64, linalg_library: str = "cusolver"):
        from cr_itd_v2.forward.materials import permittivity, refractive_index
        from cr_itd_v2.forward.pupil import pupil_quadrature
        from cr_itd_v2.forward.contracts import ACTIVE_WELL_ORDER

        if stack.layer_gap_um != 0.0:
            raise NotImplementedError("voxel3d stacks touching layers only (layer_gap_um == 0)")
        if tuple(ACTIVE_WELL_ORDER) != WELL_ORDER:
            raise RuntimeError("well order drift against the reference")
        self.stack = stack
        self.n_layers = int(stack.patterned_layer_count)
        self.layer_h = tuple(float(h) for h in stack.patterned_heights_um)
        self.device = torch.device(device)
        self.wavelengths_nm = tuple(float(w) for w in wavelengths_nm)
        self.rays = tuple(pupil_quadrature(pupil_spec) if rays is None else rays)
        self.chunk = chunk
        self.cdt = dtype
        if self.device.type == "cuda" and linalg_library:
            # One backend for every batched solve/inv/LU.  torch's default heuristic mixes
            # MAGMA and cuSOLVER by op and batch size, and on torch 2.6+cu118 that mix races
            # with the caching allocator: the first value+gradient of a fresh forward was clean,
            # every later one all-NaN; forcing either library alone removed it (4/4 clean).
            torch.backends.cuda.preferred_linalg_library(linalg_library)
        self.rdt = torch.float64 if dtype == torch.complex128 else torch.float32
        self.verbose_guard = verbose_guard
        self.guard_events: list[dict] = []
        self._perm = permittivity
        self.orders = tuple(int(o) for o in stack.fourier_orders)
        ox, oy = self.orders
        self.n_orders = (2 * ox + 1) * (2 * oy + 1)
        self.shape = tuple(int(s) for s in stack.density_shape)

        n_super = refractive_index(stack.superstrate_material, 0.55)
        self.solve_spec = []
        for li, wl_nm in enumerate(self.wavelengths_nm):
            wl = wl_nm / 1000.0
            n_s = refractive_index(stack.superstrate_material, wl)
            n_s = float(n_s.real) if isinstance(n_s, complex) else float(n_s)
            for ri, ray in enumerate(self.rays):
                s_in = float(pupil_spec.reference_index) * math.sin(ray.theta_reference_rad) / n_s
                if not 0 <= s_in < 1:
                    raise ValueError("pupil ray cannot refract into the superstrate")
                self.solve_spec.append((li, ri, wl, math.asin(s_in), float(ray.phi_rad)))
        del n_super
        self.ray_weight = torch.tensor([r.weight for r in self.rays], dtype=self.rdt, device=self.device)

        # material values per wavelength (torcwa build_permittivity / add_*_layer)
        cdt = self.cdt
        self.eps_bg = torch.tensor([complex(permittivity(stack.fill_material, w / 1000)) for w in self.wavelengths_nm],
                                   dtype=cdt, device=self.device)
        self.eps_design = torch.tensor([complex(permittivity(stack.design_material, w / 1000)) for w in self.wavelengths_nm],
                                       dtype=cdt, device=self.device)

        # Toeplitz gather indices (torcwa _material_conv)
        order_x = torch.arange(-ox, ox + 1, device=self.device)
        order_y = torch.arange(-oy, oy + 1, device=self.device)
        gx, gy = torch.meshgrid(order_x, order_y, indexing="ij")
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        self.conv_ix = (gx[:, None] - gx[None, :]) % self.shape[0]
        self.conv_iy = (gy[:, None] - gy[None, :]) % self.shape[1]

        # detector quadrants (torcwa_local._quadrant_masks, period == pitch branch)
        sx, sy = stack.detector_samples
        lx, ly = stack.period_um
        ax = (torch.arange(sx, device=self.device, dtype=self.rdt) + 0.5) * (lx / sx)
        ay = (torch.arange(sy, device=self.device, dtype=self.rdt) + 0.5) * (ly / sy)
        if stack.mosaic_pitch_um != 0.0:
            raise NotImplementedError("supercell mosaic pitch is not used by the r1a stack")
        low_x = (ax < lx / 2).reshape(-1, 1)
        low_y = (ay < ly / 2).reshape(1, -1)
        masks = {"R": low_x & low_y, "G2": ~low_x & low_y, "G1": low_x & ~low_y, "B": ~low_x & ~low_y}
        self.masks = torch.stack([masks[w].to(self.rdt) for w in WELL_ORDER])
        self.area_element = (lx / sx) * (ly / sy)

        self._geo_cache: dict[tuple, SolveGeometry] = {}
        # warm-start eigenvectors per (slot tag, solve, nudge) and the adaptive policy per slot
        self._eig_cache: dict[tuple, torch.Tensor] = {}
        self._warm_policy: dict[str, dict] = {}
        self.warm_blocked: set[str] = set()   # slots the caller keeps on full eig (e.g. low beta)
        # c128 activations of the post-eig algebra are ~130 MB per solve; 4 seeds x 144 solves
        # exhausted 80 GB (measured).  Checkpointing that part alone cuts it ~5x.
        self.checkpoint_finish = True
        self.cache_tag = "opt"   # slot tag used by response(); responses() takes one tag per density

    # ------------------------------------------------------------------ geometry
    def geometry(self, indices: Sequence[int], nudges: Sequence[float]) -> SolveGeometry:
        key = (tuple(indices), tuple(nudges))
        if key not in self._geo_cache:
            solves = [(*self.solve_spec[i][:2], self.solve_spec[i][2], self.solve_spec[i][3],
                       self.solve_spec[i][4], nudge) for i, nudge in zip(indices, nudges)]
            self._geo_cache[key] = build_geometry(
                solves, stack=self.stack, orders=self.orders,
                detector_samples=self.stack.detector_samples, period=self.stack.period_um,
                device=self.device,
                eps_in_fn=lambda wl: self._perm(self.stack.superstrate_material, wl),
                eps_out_fn=lambda wl: self._perm(self.stack.substrate_material, wl),
                cdt=self.cdt)
        return self._geo_cache[key]

    # ------------------------------------------------------------------ physics
    def _econv(self, densities: torch.Tensor):
        """E = conv(eps) and E^-1 per (density, wavelength), exactly as torcwa builds them.

        ``densities`` is [D, H, W]; rows are density-major, so (d, lam) is row d * L + lam.
        """
        rho = densities.to(self.cdt)
        eps_map = (self.eps_bg[None, :, None, None]
                   + rho[:, None] * (self.eps_design - self.eps_bg)[None, :, None, None])
        spec = (torch.fft.fft2(eps_map) / (self.shape[0] * self.shape[1])).reshape(-1, *self.shape)
        e = torch.complex(spec.real[:, self.conv_ix, self.conv_iy], spec.imag[:, self.conv_ix, self.conv_iy])
        return e, torch.linalg.inv(e)

    def _p(self, einv_all, g: SolveGeometry):
        n = self.n_orders
        ei = einv_all[g.lam_index]
        kxc, kyc = g.kx[:, :, None], g.ky[:, :, None]
        kxr, kyr = g.kx[:, None, :], g.ky[:, None, :]
        eye = torch.eye(n, dtype=ei.dtype, device=ei.device).expand_as(ei)
        return torch.cat((torch.cat((kxc * ei * kyr, eye - kxc * ei * kxr), -1),
                          torch.cat((kyc * ei * kyr - eye, -kyc * ei * kxr), -1)), -2)

    def _layer(self, e_all, einv_all, g: SolveGeometry):
        """P and the layer matrix A = P Q whose eigenpairs give the layer modes."""
        e = e_all[g.lam_index]
        kx, ky = g.kx, g.ky
        p = self._p(einv_all, g)
        q = torch.cat((torch.cat((torch.diag_embed(-kx * ky), torch.diag_embed(kx ** 2) - e), -1),
                       torch.cat((e - torch.diag_embed(ky ** 2), torch.diag_embed(ky * kx)), -1)), -2)
        return p, p @ q

    def _layer_s(self, p, vals, x, g: SolveGeometry, h: float):
        """Symmetric single-layer S-matrix blocks (T = S11 = S22, R = S12 = S21) in the vacuum reference."""
        kz = torch.sqrt(vals)
        kz = torch.where(torch.imag(kz) < 0, -kz, kz)
        hh = torch.linalg.solve(p, x * kz[:, None, :])
        w = g.vf_inv.left(hh)
        a_blk, b_blk = x + w, x - w
        phase = torch.exp(1j * g.omega[:, None] * kz * h)
        b_phi = b_blk * phase[:, None, :]
        inv_p = torch.linalg.inv(a_blk + b_phi)
        inv_m = torch.linalg.inv(a_blk - b_phi)
        sig, dif = inv_p + inv_m, inv_p - inv_m
        eye2 = torch.eye(2 * self.n_orders, dtype=x.dtype, device=x.device).expand_as(x)
        lay11 = x @ (phase[:, :, None] * sig + dif)
        lay21 = x @ (sig + phase[:, :, None] * dif) - eye2
        return lay11, lay21

    def _finish(self, ps, valss, xs, g: SolveGeometry):
        """S11 @ E_i for a batch of solves through all layers: [b, 2N, 2].

        ``ps``/``valss``/``xs`` are [b, layers, ...], layer 0 met first by the light.
        """
        eye2 = torch.eye(2 * self.n_orders, dtype=xs.dtype, device=xs.device).expand_as(xs[:, 0])
        t, r = self._layer_s(ps[:, 0], valss[:, 0], xs[:, 0], g, self.layer_h[0])
        r1 = eye2 - g.sin12.left(r)
        rhs = torch.cat((g.sin11.left(g.e_in), g.sin12.left(t)), -1)
        sol = torch.linalg.solve(r1, rhs)
        s1v = t @ sol[..., :2]
        s12 = r + t @ sol[..., 2:]
        for li in range(1, self.n_layers):
            t, r = self._layer_s(ps[:, li], valss[:, li], xs[:, li], g, self.layer_h[li])
            rr = eye2 - s12 @ r
            sol = torch.linalg.solve(rr, torch.cat((s1v, s12 @ t), -1))
            s1v = t @ sol[..., :2]
            s12 = r + t @ sol[..., 2:]
        r2 = eye2 - g.sout21.right(s12)
        return g.sout11.left(torch.linalg.solve(r2, s1v))

    def _wells(self, out: torch.Tensor, g: SolveGeometry):
        """Quadrant fractions x exact transmission for both polarisations: [b, 4, 2], T [b, 2]."""
        n = self.n_orders
        ox, oy = self.orders
        exy = out * g.out_phase[:, :, None]
        hxy = g.vo.left(exy)
        ex, ey = exy[:, :n], exy[:, n:]
        hx, hy = hxy[:, :n], hxy[:, n:]
        ez = (g.ky[:, :, None] * hx - g.kx[:, :, None] * hy) / g.eps_out[:, None, None]
        comp = torch.stack((ex, ey, ez), 1).reshape(out.shape[0], 3, 2 * ox + 1, 2 * oy + 1, 2)
        grid = torch.einsum("bxi,bcijp,byj->bcxyp", g.phase_x, comp, g.phase_y)
        intensity = (grid.real ** 2 + grid.imag ** 2).sum(1)
        field = torch.einsum("bxyp,wxy->bwp", intensity, self.masks) * self.area_element
        total = field.sum(1, keepdim=True)
        if not bool(torch.isfinite(total.detach()).all()) or bool((total.detach() <= 0).any()):
            raise RuntimeError("detector-plane field power is not finite and positive")
        fractions = field / total
        exo, eyo = out[:, :n], out[:, n:]
        c = g.coef[:, :, :, None]
        op = c[:, 0] * exo + c[:, 1] * eyo
        os_ = c[:, 2] * exo + c[:, 3] * eyo
        trans = (op.real ** 2 + op.imag ** 2 + os_.real ** 2 + os_.imag ** 2).sum(1)
        return fractions * trans[:, None, :], trans

    def _row_geometry(self, pairs, nudges):
        """Geometry of (slot, solve) groups plus the layer-row view (kx, ky, lam_index) for the eigenproblems."""
        base = self.geometry([b for _, b in pairs], nudges)
        nl, n_wl = self.n_layers, len(self.wavelengths_nm)
        slots = torch.as_tensor([d for d, _ in pairs], dtype=base.lam_index.dtype, device=base.lam_index.device)
        layer = torch.arange(nl, dtype=slots.dtype, device=slots.device)
        lam_rows = ((slots[:, None] * nl + layer[None, :]) * n_wl + base.lam_index[:, None]).reshape(-1)
        rows = SimpleNamespace(kx=base.kx.repeat_interleave(nl, 0), ky=base.ky.repeat_interleave(nl, 0),
                               lam_index=lam_rows)
        return base, rows

    def reset_warm_start(self) -> None:
        self._eig_cache.clear()
        self._warm_policy.clear()

    def _warm_allowed(self, tag: str) -> bool:
        if tag in self.warm_blocked:
            return False
        pol = self._warm_policy.get(tag)
        return pol is None or pol["ok_rate"] >= eigmod.WarmStart.min_ok_rate \
            or pol["since_try"] >= eigmod.WarmStart.retry_every

    def _chunk_geo(self, pairs, nudges, s, e):
        key = (tuple(pairs[s:e]), tuple(nudges[s:e]))
        if key not in self._geo_cache:
            self._geo_cache[key] = self._row_geometry(pairs[s:e], nudges[s:e])
        return self._geo_cache[key]

    def _chunk_eig(self, e_all, einv_all, rows):
        with torch.no_grad():
            _, a = self._layer(e_all, einv_all, rows)
            return eigmod.eig_with_warm_start(a, None, None)

    def _chunk_wells(self, e_all, einv_all, g, rows, vals, x, n_g, attach):
        p, a = self._layer(e_all, einv_all, rows)
        if attach:
            vals, x = eig_attach(a, vals, x)
        nl = self.n_layers
        pp, vals, x = (t.reshape(n_g, nl, *t.shape[1:]) for t in (p, vals, x))
        return self._wells(self._finish(pp, vals, x, g), g)

    def _run(self, pairs, nudges, e_all, einv_all, tags=None, tally=None, grad=None):
        """Wells [n, 4, 2] and transmission [n, 2] for (slot, solve) groups, in order.

        Memory is bounded per chunk of groups: eigenpairs are computed chunk by chunk (no graph); for
        differentiated slots they are parked in host memory and each chunk's algebra is recomputed in
        backward (one chunk's graph alive at a time).  Warm start is not used.
        """
        chunk = self.chunk or len(pairs)
        want = [torch.is_grad_enabled() and (grad is None or bool(grad[d])) for d, _ in pairs]
        segs: list[list] = []
        for i, w in enumerate(want):
            if not segs or segs[-1][2] != w or segs[-1][1] - segs[-1][0] >= chunk:
                segs.append([i, i, w])
            segs[-1][1] = i + 1
        gsegs = [(s, e) for s, e, w in segs if w]
        res = {}
        for s, e, w in segs:
            if not w:
                g, rows = self._chunk_geo(pairs, nudges, s, e)
                with torch.no_grad():
                    vals, x = self._chunk_eig(e_all, einv_all, rows)
                    res[(s, e)] = self._chunk_wells(e_all, einv_all, g, rows, vals, x, e - s, False)
        if gsegs:
            wl, tr = _ChunkedGrad.apply(self, pairs, nudges, gsegs, e_all, einv_all)
            o = 0
            for s, e in gsegs:
                res[(s, e)] = (wl[o:o + e - s], tr[o:o + e - s])
                o += e - s
        order = [(s, e) for s, e, _ in segs]
        return torch.cat([res[k][0] for k in order]), torch.cat([res[k][1] for k in order])

    # ------------------------------------------------------------------ public
    def response(self, density: torch.Tensor) -> torch.Tensor:
        """Tabs[well, wavelength] in float64 -- the reference's return value."""
        if density.ndim == 2:
            density = density[None]
        return self.responses(density[None], tags=(self.cache_tag,))[0]

    def responses(self, densities: torch.Tensor, tags=None, grad=None) -> torch.Tensor:
        """Tabs[density, well, wavelength] for D densities in one batch.

        Every (density, wavelength, ray) eigenproblem goes into the same
        launches, so several seeds, or a grey design and its binarised twin,
        share one eigensolver call instead of paying one each.  Each density
        is solved exactly as response() would solve it alone (no coupling
        between slots; tests/test_batching.py holds this).  ``tags`` name each
        density's warm-start slot, e.g. ("s0", "s0.bin", "s1", "s1.bin");
        ``grad`` (one bool per density) marks the densities whose gradient is
        wanted, the rest share the eigensolver call without entering the graph.
        """
        if densities.ndim != 4 or densities.shape[1] != self.n_layers or tuple(densities.shape[2:]) != self.shape:
            raise ValueError("densities must be [D, n_layers, *density_shape]")
        if densities.dtype not in (torch.float32, torch.float64):
            raise ValueError("densities must be float32/64")
        n_d = densities.shape[0]
        tags = tuple(tags) if tags is not None else tuple(f"slot{d}" for d in range(n_d))
        if len(tags) != n_d or len(set(tags)) != n_d:
            raise ValueError("one distinct tag per density")
        if grad is not None and len(grad) != n_d:
            raise ValueError("one grad flag per density")
        e_all, einv_all = self._econv(densities.reshape(-1, *self.shape))
        n_solves = len(self.solve_spec)
        pairs = [(d, b) for d in range(n_d) for b in range(n_solves)]
        main = [float(self.stack.rayleigh_frequency_nudge)] * len(pairs)
        tally: dict[str, list[int]] = {}
        wells, trans = self._run(pairs, main, e_all, einv_all, tags, tally, grad)
        for tag in tags:
            if tag in tally:
                ok, tried = tally[tag]
                self._warm_policy[tag] = dict(ok_rate=ok / tried, since_try=0)
            elif tag in self._warm_policy and eigmod.WarmStart.enabled:
                self._warm_policy[tag]["since_try"] += 1
        bad = (~torch.isfinite(trans.detach())) | (trans.detach() > 1.0 + ENERGY_GUARD_EPS)
        if bool(bad.any()):
            wells = self._guarded(pairs, wells, trans, bad, e_all, einv_all, grad)
        return torch.stack([self._aggregate(wells[d * n_solves:(d + 1) * n_solves]) for d in range(n_d)])

    def _guarded(self, pairs, wells, trans, bad, e_all, einv_all, grad=None):
        """Reference energy guard: re-solve only failing (solve, polarisation) at alternate nudges."""
        idx_bad = torch.nonzero(bad.any(1)).flatten().tolist()
        chosen: dict[tuple[int, int], float] = {}
        with torch.no_grad():
            for nudge in ENERGY_GUARD_RETRY_NUDGES:
                pending = [(r, p) for r in idx_bad for p in range(2)
                           if bool(bad[r, p]) and (r, p) not in chosen]
                if not pending:
                    break
                rs = sorted({r for r, _ in pending})
                _, tr = self._run([pairs[r] for r in rs], [nudge] * len(rs), e_all, einv_all)
                for row, r in enumerate(rs):
                    for p in range(2):
                        if (r, p) in pending and bool(torch.isfinite(tr[row, p])) and float(tr[row, p]) <= 1.0 + ENERGY_GUARD_EPS:
                            chosen[(r, p)] = nudge
                            d, b = pairs[r]
                            event = dict(density=d, wavelength_um=self.solve_spec[b][2], ray=self.solve_spec[b][1],
                                         polarization="xy"[p], transmission=float(trans[r, p]),
                                         nudge_used=nudge, transmission_retry=float(tr[row, p]))
                            self.guard_events.append(event)
                            if self.verbose_guard:
                                print("ENERGY-GUARD retry: " + ", ".join(f"{k}={v}" for k, v in event.items()), flush=True)
        unresolved = [(r, p) for r in idx_bad for p in range(2) if bool(bad[r, p]) and (r, p) not in chosen]
        if unresolved:
            r, p = unresolved[0]
            raise RuntimeError(f"energy guard: solve {self.solve_spec[pairs[r][1]]} (density {pairs[r][0]}) pol "
                               f"{'xy'[p]} exceeds {1 + ENERGY_GUARD_EPS} at every retry nudge {ENERGY_GUARD_RETRY_NUDGES}")
        # Differentiable re-solve of the replacements, then per-(solve, pol) assembly.
        combos = sorted({(r, n) for (r, _), n in chosen.items()})
        rw, _ = self._run([pairs[r] for r, _ in combos], [n for _, n in combos], e_all, einv_all, grad=grad)
        pos = {c: i for i, c in enumerate(combos)}
        cols = []
        for p in range(2):
            src = wells[:, :, p]
            rows = [rw[pos[(r, chosen[(r, p)])], :, p] if (r, p) in chosen else None for r in range(wells.shape[0])]
            if any(x is not None for x in rows):
                src = torch.stack([x if x is not None else src[r] for r, x in enumerate(rows)])
            cols.append(src)
        return torch.stack(cols, -1)

    def _aggregate(self, wells: torch.Tensor) -> torch.Tensor:
        n_rays = len(self.rays)
        per = wells.sum(-1).reshape(len(self.wavelengths_nm), n_rays, 4)
        tabs = (per * (self.ray_weight / 2.0)[None, :, None]).sum(1).T
        tabs = tabs.to(torch.float64)
        if not bool(torch.isfinite(tabs.detach()).all()):
            raise RuntimeError("local spectral response contains non-finite values")
        if bool((tabs.detach() < 0).any()):
            raise RuntimeError("local spectral response contains negative power")
        return tabs
