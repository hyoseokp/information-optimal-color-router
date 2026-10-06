"""voxel3d CLASSIC optimiser: TiO2/air voxel colour router, peak x purity objective, diagonal (x<->y) symmetric, AM connectivity gate.

Objective (no I_img in the loss): bands B=420/450/470, G=510/540/570, R=600/635/670 nm (9 training wavelengths, 3 per band,
no sample in the 480-500 / 570-590 gaps).  Colour k: T_k = own-well power (G = G1+G2), tot = sum of 4 wells.
peak_k = mean_band T_k, purity_k = mean_band T_k/tot, s_k = log(peak_k*purity_k), J = sum_k s_k + softmin_tau(s_k).
The schedule figure of merit is F = mean_k peak_k*purity_k (positive, <=1).  I_img is only logged.


Light enters from glass (superstrate SiO2_nk); layer 0 is met first and rests on the glass; then air gap, detector.
Support chain runs 0 -> L-1:
  grey   : rho~_i = rho_i * clamp(2 * dilate(rho~_{i-1}), 0, 1)
  binary : rho~_i = (rho_i > .5) & dilate_bin(rho~_{i-1})        (exact; checked independently with torch.roll)
dilate = circular 3x3 max-pool applied --reach times.  Layer 0 rests on the glass (no gate).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cr_itd_v2.forward import DEFAULT_R1A_PUPIL, DEFAULT_R1A_STACK  # noqa: E402
from cr_itd_v2.forward.contracts import PupilRay  # noqa: E402
from voxel3d.common import RAY_RULES, TRAIN_WAVELENGTHS_NM, gaussian_filter_periodic, i_img, load_context, project  # noqa: E402
from voxel3d import eig as eigmod, materials_siefke  # noqa: E402
from voxel3d.forward import FastPupilForward  # noqa: E402

WL = np.array(TRAIN_WAVELENGTHS_NM)
ORDER = ("R", "G2", "G1", "B")


def dilate(x, reach):
    for _ in range(reach):
        p = F.pad(x[None, None], (1, 1, 1, 1), mode="circular")
        x = F.max_pool2d(p, 3, 1)[0, 0]
    return x


def reach_of(i):
    """One 10 nm voxel of lateral reach every second layer (slope <= 0.5)."""
    return 1 if i % 2 == 0 else 0


def gate_grey(rhos, reach=None):
    out = [None] * len(rhos)
    out[0] = rhos[0]
    for i in range(1, len(rhos)):
        out[i] = rhos[i] * (2.0 * dilate(out[i - 1], reach_of(i))).clamp(0, 1)
    return torch.stack(out)


def gate_binary(rhos, reach=None):
    b = [(r.detach() > 0.5).to(r.dtype) for r in rhos]
    out = [None] * len(b)
    out[0] = b[0]
    for i in range(1, len(b)):
        out[i] = b[i] * dilate(out[i - 1], reach_of(i))
    return torch.stack(out)


def unsupported(stackb, reach=None):
    """Independent check: voxels with no solid voxel within the (2*reach+1)^2 square below (periodic)."""
    n = 0
    for i in range(1, stackb.shape[0]):
        reach = reach_of(i)
        sup = torch.zeros_like(stackb[i])
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                sup = torch.maximum(sup, torch.roll(stackb[i - 1], (dy, dx), (0, 1)))
        n += int(((stackb[i] > 0.5) & (sup < 0.5)).sum())
    return n


def trap(y, m):
    x, y = WL[m], y[m]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) if len(x) > 1 else float(y.sum())


def crosstalk(tabs):
    """Old-CR Eq.(2) on the 9 training wavelengths (trapezoid). tabs [4, 9] in R,G2,G1,B order."""
    t = np.asarray(tabs, dtype=np.float64)
    R, G, B = t[0], t[1] + t[2], t[3]
    allm = np.ones_like(WL, bool)
    cr = trap(R, WL <= 600) / max(trap(R, allm), 1e-30)
    cg = trap(G, (WL <= 500) | (WL >= 600)) / max(trap(G, allm), 1e-30)
    cb = trap(B, WL >= 500) / max(trap(B, allm), 1e-30)
    k = lambda c: int(np.argmin(np.abs(WL - c)))  # noqa: E731
    eta = dict(R=float(t[0, k(650)]), G=float(t[1, k(550)] + t[2, k(550)]), B=float(t[3, k(450)]))
    return dict(C_R=cr, C_G=cg, C_B=cb, C_mean=(cr + cg + cb) / 3, eta=eta)


BANDS = dict(B=[0, 1, 2], G=[3, 4, 5], R=[6, 7, 8])


def classic(tabs, tau):
    """tabs [4, 9] in R,G2,G1,B order (torch, differentiable) -> J, F and per-colour peak/purity."""
    tot = tabs.sum(0)
    T = dict(R=tabs[0], G=tabs[1] + tabs[2], B=tabs[3])
    peak, pur, s = {}, {}, []
    for k in ("R", "G", "B"):
        b = BANDS[k]
        peak[k] = T[k][b].mean()
        pur[k] = (T[k][b] / tot[b]).mean()
        s.append(torch.log(peak[k] * pur[k]))
    s = torch.stack(s)
    J = s.sum() - tau * torch.logsumexp(-s / tau, 0)
    F = sum(peak[k] * pur[k] for k in peak) / 3.0
    wells = dict(R=tabs[0][BANDS["R"]].mean(), G2=tabs[1][BANDS["G"]].mean(), G1=tabs[2][BANDS["G"]].mean(), B=tabs[3][BANDS["B"]].mean())
    return J, F, dict(peak={k: float(v) for k, v in peak.items()}, purity={k: float(v) for k, v in pur.items()},
                      peak_well={k: float(v) for k, v in wells.items()}, F=float(F), J=float(J),
                      sym_err=float((tabs[1] - tabs[2]).abs().max()))


def sym(th):
    return 0.5 * (th + th.transpose(-1, -2))


def gauss3d(th, sigma):
    """3D Gaussian on [L,H,W]: periodic in x/y, replicate (zero-gradient) padding in z at the glass and air faces.
    Same sigma along x and y, so a transpose-symmetric input stays transpose-symmetric."""
    r = max(1, int(np.ceil(3.0 * sigma)))
    t = torch.arange(-r, r + 1, device=th.device, dtype=th.dtype)
    k = torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    k = k / k.sum()
    y = th[None, None]
    y = F.conv3d(F.pad(y, (r, r, 0, 0, 0, 0), mode="circular"), k.view(1, 1, 1, 1, -1))
    y = F.conv3d(F.pad(y, (0, 0, r, r, 0, 0), mode="circular"), k.view(1, 1, 1, -1, 1))
    y = F.conv3d(F.pad(y, (0, 0, 0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1, 1))
    return y[0, 0]


def zup(fs, sub):
    """Linear z-interpolation of C control fields [C,H,W] (centres (i+.5)h) to C*sub RCWA layers (clamped ends)."""
    if sub == 1:
        return list(fs)
    C = len(fs)
    out = []
    for j in range(C * sub):
        t = (j + 0.5) / sub - 0.5
        i0 = min(max(int(np.floor(t)), 0), C - 1)
        i1 = min(i0 + 1, C - 1)
        w = min(max(t - i0, 0.0), 1.0) if C > 1 else 0.0
        out.append(fs[i0] * (1 - w) + fs[i1] * w)
    return out


def build(args):
    h, L = args.h / args.sub, args.layers * args.sub
    kw = dict(density_shape=(200, 200))
    if args.small:
        kw = dict(density_shape=(16, 16), fourier_orders=(2, 2), detector_samples=(8, 8))
    stack = dataclasses.replace(DEFAULT_R1A_STACK, superstrate_material="SiO2_nk", substrate_material="air", design_material=materials_siefke.NAME,
                                design_fill_material="air", design_height_um=h,
                                layer_heights_um=(h,) * L if L > 1 else (), detector_z_um=L * h + args.gap_um, **kw)
    radial, count = (1, 1) if args.rays == 1 else RAY_RULES[args.rays]
    pupil = dataclasses.replace(DEFAULT_R1A_PUPIL, radial_order=radial, azimuth_count=count,
                                chief_ray_angle_deg=args.cra, chief_ray_azimuth_deg=args.az)
    return stack, pupil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=100)
    ap.add_argument("--h", type=float, default=0.01)
    ap.add_argument("--gap-um", type=float, default=0.45)
    ap.add_argument("--rays", type=int, default=1, help="1 = single normal-incidence ray (weight 1); 6/8/16 = pupil ensembles")
    ap.add_argument("--reach", type=int, default=3)
    ap.add_argument("--sub", type=int, default=1, help="RCWA layers per control layer (z-interpolated)")
    ap.add_argument("--init-state", default=None, help="state.pt whose theta/m/v/beta start this run")
    ap.add_argument("--sigma-nm", type=float, default=50.0)
    ap.add_argument("--beta0", type=float, default=2.0)
    ap.add_argument("--betamax", type=float, default=1024.0)
    ap.add_argument("--bstep", type=float, default=1.3)
    ap.add_argument("--smin", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lrbeta", type=float, default=0.4)
    ap.add_argument("--lr-h", type=float, default=1.0, help="height in um used in the lr rule (total stack height)")
    ap.add_argument("--warm-steps", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--bineval", type=int, default=10)
    ap.add_argument("--outer", type=int, default=600)
    ap.add_argument("--tuned", default="")
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--cra", type=float, default=0.0)
    ap.add_argument("--az", type=float, default=0.0, help="chief-ray azimuth; 45 makes the ray set x<->y symmetric")
    ap.add_argument("--tau", type=float, default=0.05)
    args = ap.parse_args()

    materials_siefke.install()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned and Path(args.tuned).exists() else {}
    eigmod.EigConfig.backend = tuned.get("eig_backend", "gpu") if dev.type == "cuda" else "cpu"
    eigmod.EigConfig.cpu_workers = int(tuned.get("eig_cpu_workers", 4))
    eigmod.WarmStart.enabled = False
    out = Path(args.out)
    (out / "steps").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(vars(args), tuned=tuned), indent=2))

    stack, pupil = build(args)
    context = load_context()
    fwd = FastPupilForward(stack=stack, pupil_spec=pupil, wavelengths_nm=TRAIN_WAVELENGTHS_NM, device=dev,
                           chunk=args.chunk, dtype=torch.complex128,
                           rays=(PupilRay(0.0, 0.0, 0.0, 0.0, 1.0),) if args.rays == 1 else None)
    fwd.checkpoint_finish = True
    shape = tuple(stack.density_shape)
    sigma_px = args.sigma_nm / (stack.period_um[0] * 1000.0 / shape[0])
    L = args.layers
    h_um = args.h

    st_path = out / "state.pt"
    if st_path.exists():
        st = torch.load(st_path, map_location=dev)
        theta, m, v = st["theta"], st["m"], st["v"]
        k0, beta, last, hist, best_bin = st["k"] + 1, st["beta"], st["last"], st["hist"], st["best_bin"]
        print(f"[run] resumed at step {k0}, beta {beta:g}", flush=True)
    elif args.init_state:
        st = torch.load(args.init_state, map_location=dev)
        theta, m, v, beta = st["theta"], st["m"], st["v"], st["beta"]
        k0, last, hist, best_bin = 0, 0, [], -1e30
        print(f"[run] init from {args.init_state}: theta {tuple(theta.shape)}, beta {beta:g}", flush=True)
    else:
        gen = torch.Generator().manual_seed(args.seed)
        a = (0.5 + args.noise * (2 * torch.rand((L,) + shape, generator=gen, dtype=torch.float64) - 1)).clamp(0, 1)
        theta = torch.triu(a) + torch.triu(a, 1).transpose(-1, -2)
        theta = theta.to(dev)
        m, v = torch.zeros_like(theta), torch.zeros_like(theta)
        k0, beta, last, hist, best_bin = 0, args.beta0, 0, [], -1e30

    def rhos_of(th, b):
        ths = sym(th)
        f = gauss3d(ths, sigma_px)
        return [project(f[i], b).clamp(0, 1) for i in range(L)]

    def tabs_i(dens, grad):
        with torch.set_grad_enabled(grad):
            tabs = fwd.responses(dens[None], tags=("g" if grad else "b",), grad=[grad])[0]
            J, F, info = classic(tabs, args.tau)
            with torch.no_grad():
                info["I_img_ray"] = args.rays
                info["I_img"] = i_img(tabs.detach(), context)
            return tabs, J, F, info

    t_run = time.time()
    for k in range(k0, args.outer):
        if (out / "STOP").exists():
            print("[run] STOP file", flush=True)
            break
        t0 = time.time()
        hI = [h_["F_grey"] for h_ in hist if h_["step"] >= last]
        brk = False
        if k > 0:
            if (k - last) >= args.smin and len(hI) >= 11 and (hI[-1] - hI[-11]) < 0.0005 * abs(hI[-11]) and beta < args.betamax:
                beta = min(beta * args.bstep, args.betamax)
                last_new = k
                print(f"[bstep] step {k}: grey saturated (10-step {hI[-11]:.5f}->{hI[-1]:.5f}), beta -> {beta:g}", flush=True)
                last = last_new
            elif beta >= args.betamax and (k - last) >= 10 and len(hI) >= 11 and (hI[-1] - hI[-11]) < 0.0005 * abs(hI[-11]):
                print(f"[bstep] step {k}: beta max (1024), 10-step gain {hI[-11]:.5f}->{hI[-1]:.5f} < 0.05 %, stop", flush=True)
                break
        beta_changed = bool(hist) and abs(hist[-1]["beta"] - beta) > 1e-12
        do_bin = (k % args.bineval == 0) or beta_changed

        th = theta.clone().requires_grad_(True)
        rs = rhos_of(th, beta)
        dens = gate_grey(rs, args.reach)
        tabs, obj, Fg, info_g = tabs_i(dens, True)
        obj.backward()
        g = th.grad.detach()
        ct_g = crosstalk(tabs.detach().cpu().numpy())
        t_fb = time.time() - t0

        rec = dict(step=k, J_grey=float(obj), F_grey=float(Fg), info_grey=info_g, I_img_grey=info_g["I_img"], F_bin=None, info_bin=None, I_img_bin=None, best_bin=best_bin, beta=beta, ct_grey=ct_g, ct_bin=None,
                   unsupported_bin=None, t_fb=t_fb)
        if do_bin:
            with torch.no_grad():
                bstack = gate_binary([r.detach() for r in rs], args.reach)
                tb, Jb, Fb, info_b = tabs_i(bstack, False)
            rec["F_bin"] = float(Fb)
            rec["info_bin"] = info_b
            rec["I_img_bin"] = info_b["I_img"]
            rec["ct_bin"] = crosstalk(tb.cpu().numpy())
            rec["unsupported_bin"] = unsupported(bstack, args.reach)
            if rec["F_bin"] > best_bin:
                best_bin = rec["F_bin"]
                np.savez(out / "best.npz", stack=bstack.cpu().numpy().astype(np.uint8), tabs=tb.cpu().numpy(),
                         F_bin=best_bin, I_img_bin=rec["I_img_bin"], step=k, beta=beta, ct=json.dumps(rec["ct_bin"]), info=json.dumps(info_b))
            rec["best_bin"] = best_bin
            np.savez_compressed(out / "steps" / f"bin_{k:04d}.npz", stack=bstack.cpu().numpy().astype(np.uint8),
                                tabs=tb.cpu().numpy(), F_bin=rec["F_bin"], I_img_bin=rec["I_img_bin"], step=k, beta=beta)

        lr = min(args.lr, (args.lrbeta / args.lr_h) / beta) * min(1.0, (k + 1) / args.warm_steps)
        with torch.no_grad():
            m = torch.where(m * g < 0, torch.zeros_like(m), m)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            theta = (theta + lr * (m / (1 - 0.9 ** (k + 1))) / ((v / (1 - 0.999 ** (k + 1))).sqrt() + 1e-12)).clamp(0, 1)
        rec.update(lr=lr, t_step=time.time() - t0)
        hist.append(rec)
        np.savez_compressed(out / "steps" / f"theta_{k:04d}.npz", theta=theta.cpu().numpy().astype(np.float32),
                            tabs_grey=tabs.detach().cpu().numpy(), J_grey=float(obj), F_grey=float(Fg), I_img_grey=info_g["I_img"], beta=beta, step=k)
        (out / "history.json").write_text(json.dumps(hist))
        tmp = out / "state.tmp"
        torch.save(dict(theta=theta, m=m, v=v, k=k, beta=beta, last=last, hist=hist, best_bin=best_bin), tmp)
        tmp.replace(st_path)
        ib = f"{rec['F_bin']:.5f}" if rec["F_bin"] is not None else "-"
        pk, pu = info_g["peak"], info_g["purity"]
        print(f"step {k:3d} F(grey) {rec['F_grey']:.5f} J {rec['J_grey']:.4f} bin {ib} beta {beta:.3g} lr {lr:.4f} "
              f"pk R{pk['R']:.3f} G{pk['G']:.3f} B{pk['B']:.3f} pu R{pu['R']:.3f} G{pu['G']:.3f} B{pu['B']:.3f} "
              f"Cx {ct_g['C_mean']:.3f} Iimg {rec['I_img_grey']:.4f} sym {info_g['sym_err']:.1e} unsup {rec['unsupported_bin']} fb {t_fb:.0f}s step {rec['t_step']:.0f}s", flush=True)
    print("[run] done", flush=True)


if __name__ == "__main__":
    main()
