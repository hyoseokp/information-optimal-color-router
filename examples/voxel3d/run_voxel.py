"""voxel3d optimiser: TiO2/air multi-layer voxel colour router, paper I_img objective, AM connectivity gate.

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
from cr_itd_v2.optimization.local_objectives import evaluate_local_arm_objective  # noqa: E402
from voxel3d.common import RAY_RULES, TRAIN_WAVELENGTHS_NM, gaussian_filter_periodic, load_context, project  # noqa: E402
from voxel3d import eig as eigmod, materials_siefke  # noqa: E402
from voxel3d.forward import FastPupilForward  # noqa: E402

WL = np.array(TRAIN_WAVELENGTHS_NM)
ORDER = ("R", "G2", "G1", "B")


def dilate(x, reach):
    for _ in range(reach):
        p = F.pad(x[None, None], (1, 1, 1, 1), mode="circular")
        x = F.max_pool2d(p, 3, 1)[0, 0]
    return x


def gate_grey(rhos, reach):
    out = [None] * len(rhos)
    out[0] = rhos[0]
    for i in range(1, len(rhos)):
        out[i] = rhos[i] * (2.0 * dilate(out[i - 1], reach)).clamp(0, 1)
    return torch.stack(out)


def gate_binary(rhos, reach):
    b = [(r.detach() > 0.5).to(r.dtype) for r in rhos]
    out = [None] * len(b)
    out[0] = b[0]
    for i in range(1, len(b)):
        out[i] = b[i] * dilate(out[i - 1], reach)
    return torch.stack(out)


def unsupported(stackb, reach):
    """Independent check: voxels with no solid voxel within the (2*reach+1)^2 square below (periodic)."""
    n = 0
    for i in range(1, stackb.shape[0]):
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
    kw = {}
    if args.small:
        kw = dict(density_shape=(16, 16), fourier_orders=(2, 2), detector_samples=(8, 8))
    stack = dataclasses.replace(DEFAULT_R1A_STACK, superstrate_material="SiO2_nk", substrate_material="air", design_material=materials_siefke.NAME,
                                design_fill_material="air", design_height_um=h,
                                layer_heights_um=(h,) * L if L > 1 else (), detector_z_um=L * h + args.gap_um, **kw)
    radial, count = RAY_RULES[args.rays]
    pupil = dataclasses.replace(DEFAULT_R1A_PUPIL, radial_order=radial, azimuth_count=count)
    return stack, pupil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--gap-um", type=float, default=0.45)
    ap.add_argument("--rays", type=int, default=16)
    ap.add_argument("--reach", type=int, default=3)
    ap.add_argument("--sub", type=int, default=1, help="RCWA layers per control layer (z-interpolated)")
    ap.add_argument("--init-state", default=None, help="state.pt whose theta/m/v/beta start this run")
    ap.add_argument("--sigma-nm", type=float, default=20.0)
    ap.add_argument("--beta0", type=float, default=2.0)
    ap.add_argument("--betamax", type=float, default=256.0)
    ap.add_argument("--bstep", type=float, default=1.3)
    ap.add_argument("--smin", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lrbeta", type=float, default=0.4)
    ap.add_argument("--warm-steps", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--bineval", type=int, default=10)
    ap.add_argument("--outer", type=int, default=600)
    ap.add_argument("--tuned", default="")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--small", action="store_true")
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
                           chunk=args.chunk, dtype=torch.complex128)
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
        theta = (0.5 + args.noise * (2 * torch.rand((L,) + shape, generator=gen, dtype=torch.float64) - 1)).clamp(0, 1).to(dev)
        m, v = torch.zeros_like(theta), torch.zeros_like(theta)
        k0, beta, last, hist, best_bin = 0, args.beta0, 0, [], -1e30

    def rhos_of(th, b):
        f = [gaussian_filter_periodic(th[i], sigma_px) for i in range(L)]
        return [project(x, b).clamp(0, 1) for x in zup(f, args.sub)]

    def tabs_i(dens, grad):
        with torch.set_grad_enabled(grad):
            tabs = fwd.responses(dens[None], tags=("g" if grad else "b",), grad=[grad])[0]
            return tabs, evaluate_local_arm_objective(tabs, context, "itd").objective

    t_run = time.time()
    for k in range(k0, args.outer):
        if (out / "STOP").exists():
            print("[run] STOP file", flush=True)
            break
        t0 = time.time()
        hI = [h_["I_grey"] for h_ in hist if h_["step"] >= last]
        brk = False
        if k > 0:
            if (k - last) >= args.smin and len(hI) >= 4 and 0 <= (hI[-1] - hI[-4]) < 0.005 * abs(hI[-4]) and beta < args.betamax:
                beta = min(beta * args.bstep, args.betamax)
                last_new = k
                print(f"[bstep] step {k}: grey saturated ({hI[-4]:.4f}->{hI[-1]:.4f}), beta -> {beta:g}", flush=True)
                last = last_new
            elif beta >= args.betamax and (k - last) >= 10 and len(hI) >= 11 and 0 <= (hI[-1] - hI[-11]) < 0.01 * abs(hI[-11]):
                print(f"[bstep] step {k}: beta max, 10-step gain {hI[-11]:.4f}->{hI[-1]:.4f} < 1 %, stop", flush=True)
                break
        beta_changed = bool(hist) and abs(hist[-1]["beta"] - beta) > 1e-12
        do_bin = (k % args.bineval == 0) or beta_changed

        th = theta.clone().requires_grad_(True)
        rs = rhos_of(th, beta)
        dens = gate_grey(rs, args.reach)
        tabs, obj = tabs_i(dens, True)
        obj.backward()
        g = th.grad.detach()
        ct_g = crosstalk(tabs.detach().cpu().numpy())
        t_fb = time.time() - t0

        rec = dict(step=k, I_grey=float(obj), I_bin=None, best_bin=best_bin, beta=beta, ct_grey=ct_g, ct_bin=None,
                   unsupported_bin=None, t_fb=t_fb)
        if do_bin:
            with torch.no_grad():
                bstack = gate_binary([r.detach() for r in rs], args.reach)
                tb, ob = tabs_i(bstack, False)
            rec["I_bin"] = float(ob)
            rec["ct_bin"] = crosstalk(tb.cpu().numpy())
            rec["unsupported_bin"] = unsupported(bstack, args.reach)
            if rec["I_bin"] > best_bin:
                best_bin = rec["I_bin"]
                np.savez(out / "best.npz", stack=bstack.cpu().numpy().astype(np.uint8), tabs=tb.cpu().numpy(),
                         I_bin=best_bin, step=k, beta=beta, ct=json.dumps(rec["ct_bin"]))
            rec["best_bin"] = best_bin
            np.savez_compressed(out / "steps" / f"bin_{k:04d}.npz", stack=bstack.cpu().numpy().astype(np.uint8),
                                tabs=tb.cpu().numpy(), I_bin=rec["I_bin"], step=k, beta=beta)

        lr = min(args.lr, (args.lrbeta / h_um) / beta) * min(1.0, (k + 1) / args.warm_steps)
        with torch.no_grad():
            m = torch.where(m * g < 0, torch.zeros_like(m), m)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            theta = (theta + lr * (m / (1 - 0.9 ** (k + 1))) / ((v / (1 - 0.999 ** (k + 1))).sqrt() + 1e-12)).clamp(0, 1)
        rec.update(lr=lr, t_step=time.time() - t0)
        hist.append(rec)
        np.savez_compressed(out / "steps" / f"theta_{k:04d}.npz", theta=theta.cpu().numpy().astype(np.float32),
                            tabs_grey=tabs.detach().cpu().numpy(), I_grey=float(obj), beta=beta, step=k)
        (out / "history.json").write_text(json.dumps(hist))
        tmp = out / "state.tmp"
        torch.save(dict(theta=theta, m=m, v=v, k=k, beta=beta, last=last, hist=hist, best_bin=best_bin), tmp)
        tmp.replace(st_path)
        ib = f"{rec['I_bin']:.5f}" if rec["I_bin"] is not None else "-"
        print(f"step {k:3d} I(grey) {rec['I_grey']:.5f} bin {ib} beta {beta:.3g} lr {lr:.4f} "
              f"Cx {ct_g['C_mean']:.3f} unsup {rec['unsupported_bin']} fb {t_fb:.0f}s step {rec['t_step']:.0f}s", flush=True)
    print("[run] done", flush=True)


if __name__ == "__main__":
    main()
