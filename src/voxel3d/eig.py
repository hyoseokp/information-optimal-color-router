"""Batched non-Hermitian eigendecomposition with torcwa's stabilised gradient.

The layer eigenproblem of a lossless grating is not Hermitian in general:
complex-conjugate pairs of kz^2 (complex modes) are physical, so there is no
Cholesky/eigh shortcut.  This is the one operation that decides step time.

Forward backends
  "gpu"     torch.linalg.eig on the device tensor, optionally split over
            several CUDA streams so per-matrix solver calls overlap.
  "cpu"     the batch is copied to the host and split over a thread pool; each
            worker calls LAPACK on its slice (torch releases the GIL).
  "hybrid"  a fraction of the batch goes to each, concurrently.
The right choice is machine-specific (it depends on whether this torch build
routes CUDA eig to cuSOLVER Xgeev or to a CPU fallback, and on core count), so
bench/autotune.py measures and picks it; nothing here guesses.

Backward is torcwa.torch_eig.Eig.backward, batched: Lorentzian broadening
1e-10 on eigenvalue gaps, gradient inv(X^H) (diag(g_lambda) + conj(F) *
(X^H g_X)) X^H.  Same formula as the reference, so gradients agree with the
per-solve reference path.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import torch

BROADENING = 1e-10


class EigConfig:
    backend: str = "gpu"
    streams: int = 1
    cpu_workers: int = 4
    gpu_fraction: float = 0.5
    _pool: ThreadPoolExecutor | None = None
    _lock = threading.Lock()

    @classmethod
    def pool(cls) -> ThreadPoolExecutor:
        with cls._lock:
            if cls._pool is None or cls._pool._max_workers != cls.cpu_workers:
                cls._pool = ThreadPoolExecutor(max_workers=max(1, cls.cpu_workers))
            return cls._pool


def _device_sync(t: torch.Tensor) -> None:
    """Fence after a CUDA eigensolver call.

    On torch 2.6+cu118 (MAGMA path) torch.linalg.eig finishes writing its
    outputs on a queue the caching allocator does not track.  Without this
    fence a block freed afterwards can be handed to a new tensor while that
    write is still in flight: measured on an RTX 3060, the first
    value+gradient of a fresh forward was clean and every later one was
    all-NaN, and the NaN vanished under CUDA_LAUNCH_BLOCKING=1 or with the
    caching allocator disabled.  The fence costs one synchronisation per
    eigendecomposition batch, which is noise next to the eigensolve itself.
    """
    if t.device.type == "cuda":
        torch.cuda.synchronize(t.device)


def _eig_gpu(a: torch.Tensor, streams: int):
    if streams <= 1 or a.shape[0] < 2 or a.device.type != "cuda":
        out = torch.linalg.eig(a)
        _device_sync(a)
        return out
    chunks = torch.chunk(a, streams, dim=0)
    results = [None] * len(chunks)
    main = torch.cuda.current_stream(a.device)
    side = [torch.cuda.Stream(device=a.device) for _ in chunks]

    def work(i):
        side[i].wait_stream(main)
        with torch.cuda.stream(side[i]):
            results[i] = torch.linalg.eig(chunks[i])

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(chunks))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for s in side:
        main.wait_stream(s)
    _device_sync(a)
    return torch.cat([r[0] for r in results]), torch.cat([r[1] for r in results])


def _eig_cpu(a: torch.Tensor, workers: int):
    host = a.detach().to("cpu", non_blocking=False)
    chunks = torch.chunk(host, max(1, workers), dim=0)
    futures = [EigConfig.pool().submit(torch.linalg.eig, c) for c in chunks]
    out = [f.result() for f in futures]
    vals = torch.cat([o[0] for o in out]).to(a.device)
    vecs = torch.cat([o[1] for o in out]).to(a.device)
    return vals, vecs


def _eig_hybrid(a: torch.Tensor):
    n_gpu = int(round(a.shape[0] * EigConfig.gpu_fraction))
    if n_gpu <= 0:
        return _eig_cpu(a, EigConfig.cpu_workers)
    if n_gpu >= a.shape[0]:
        return _eig_gpu(a, EigConfig.streams)
    cpu_future = EigConfig.pool().submit(_eig_cpu, a[n_gpu:], max(1, EigConfig.cpu_workers - 1))
    gv, gx = _eig_gpu(a[:n_gpu], EigConfig.streams)
    cv, cx = cpu_future.result()
    return torch.cat((gv, cv)), torch.cat((gx, cx))


def eig_forward(a: torch.Tensor):
    backend = EigConfig.backend
    if backend == "gpu":
        return _eig_gpu(a, EigConfig.streams)
    if backend == "cpu":
        return _eig_cpu(a, EigConfig.cpu_workers)
    if backend == "hybrid":
        return _eig_hybrid(a)
    raise ValueError(f"unknown eig backend {backend!r}")


def refine_eig(a: torch.Tensor, x0: torch.Tensor, *, iters: int, tol: float,
               abandon_after: int = 3, abandon_residual: float = 1e-2):
    """Newton-type refinement of eigenpairs from a nearby matrix's eigenvectors.

    B = X^-1 A X is nearly diagonal when A moved little since X was computed;
    X <- X (I + F), F_ij = B_ij / (b_jj - b_ii), then converges quadratically
    using one GEMM and one batched LU per sweep.  It only works when the change
    is small against the eigenvalue spacing.  Measured on real zonec
    trajectories (RTX 3060, 16 solves at 540 nm): in the grey and mid phases a
    step moves A by ~2.4 % while the first-order coupling is ~30-40x the gap
    for hundreds of pairs, and the iteration diverges; near-binary (beta ~160)
    A moves ~0.2 %, coupling/gap ~0.85, and 12 sweeps reach a median residual
    of 1.6e-6 (full LAPACK eig: 4e-6).  So it is attempted adaptively and
    abandoned early, never trusted blindly.

    Acceptance per element, measured on the returned pairs:
      * relative residual ||A X - X diag(vals)||_F / ||A||_F < tol, and
      * X not collapsed: min |U_ii| / max |U_ii| of its LU above a floor
        (two columns converging to one eigenvector each have a tiny residual,
        so the residual alone cannot catch that).
    Returns (vals, vecs, ok); callers re-solve every element with ok False.
    """
    x = x0 / torch.linalg.vector_norm(x0, dim=-2, keepdim=True)
    eye = torch.eye(a.shape[-1], dtype=torch.bool, device=a.device)
    floor = 1e-12 if a.dtype == torch.complex128 else 1e-6
    norm_a = torch.linalg.matrix_norm(a)
    for it in range(iters):
        lu, piv, _ = torch.linalg.lu_factor_ex(x)
        b = torch.linalg.lu_solve(lu, piv, a @ x)
        d = b.diagonal(dim1=-2, dim2=-1)
        gap = d.unsqueeze(-2) - d.unsqueeze(-1)
        off = b.masked_fill(eye, 0)
        f = torch.where(eye, torch.zeros_like(off), off / torch.where(eye, torch.ones_like(gap), gap))
        x = x + x @ f
        x = x / torch.linalg.vector_norm(x, dim=-2, keepdim=True)
        rel_off = torch.linalg.matrix_norm(off) / torch.linalg.matrix_norm(b).clamp_min(1e-300)
        if it + 1 == abandon_after and not bool((torch.nan_to_num(rel_off, nan=1e30) < abandon_residual).any()):
            return d, x, torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
        if bool((rel_off < tol).all()):
            break
    lu, piv, info = torch.linalg.lu_factor_ex(x)
    b = torch.linalg.lu_solve(lu, piv, a @ x)
    vals = b.diagonal(dim1=-2, dim2=-1).clone()
    res = torch.linalg.matrix_norm(a @ x - x * vals.unsqueeze(-2)) / norm_a
    udiag = lu.diagonal(dim1=-2, dim2=-1).abs()
    complete = (udiag.amin(-1) / udiag.amax(-1)) > floor
    ok = (res < tol) & complete & torch.isfinite(res) & (info == 0)
    return vals, x, ok


class WarmStart:
    """Adaptive warm-start policy and statistics (see refine_eig for evidence)."""

    enabled: bool = False
    iters: int = 12
    tol_c64: float = 1e-5
    tol_c128: float = 1e-11
    min_ok_rate: float = 0.5     # keep refining a batch only while at least this share converges
    retry_every: int = 5         # otherwise re-test the batch every N calls
    chunk: int | None = 144      # rows refined at once (each sweep holds ~8 matrices per row)
    stats = {"refined": 0, "fallback": 0, "cold": 0}
    last_ok_rate: float | None = None
    last_ok: torch.Tensor | None = None       # per attempted row, for per-slot policies


def eig_with_warm_start(a: torch.Tensor, warm: torch.Tensor | None, rows: torch.Tensor | None = None):
    """Eigenpairs of a; ``warm`` holds starting eigenvectors for a[rows] (all rows if None).

    Rows without a warm start, and warm rows that fail acceptance, get a full
    eigendecomposition in one batched call.
    """
    WarmStart.last_ok_rate = None
    WarmStart.last_ok = None
    if warm is None or not WarmStart.enabled:
        WarmStart.stats["cold"] += a.shape[0]
        return eig_forward(a)
    if rows is None:
        rows = torch.arange(a.shape[0], device=a.device)
    tol = WarmStart.tol_c128 if a.dtype == torch.complex128 else WarmStart.tol_c64
    step = WarmStart.chunk or rows.numel()
    got = [refine_eig(a[rows[i:i + step]], warm[i:i + step], iters=WarmStart.iters, tol=tol)
           for i in range(0, rows.numel(), step)]
    vals_r, vecs_r, ok = (torch.cat([r[j] for r in got]) for j in range(3))
    del got
    WarmStart.stats["refined"] += int(ok.sum())
    WarmStart.stats["fallback"] += int((~ok).sum())
    WarmStart.stats["cold"] += a.shape[0] - rows.numel()
    WarmStart.last_ok_rate = float(ok.float().mean())
    WarmStart.last_ok = ok
    need = torch.ones(a.shape[0], dtype=torch.bool, device=a.device)
    good = rows[ok]
    need[good] = False
    vals = torch.empty(a.shape[:-1], dtype=a.dtype, device=a.device)
    vecs = torch.empty_like(a)
    vals[good] = vals_r[ok]
    vecs[good] = vecs_r[ok]
    bad = torch.nonzero(need).flatten()
    if bad.numel():
        v2, x2 = eig_forward(a[bad])
        vals[bad] = v2
        vecs[bad] = x2
    return vals, vecs


class BatchedEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, warm=None, rows=None):
        vals, vecs = eig_with_warm_start(a, warm, rows)
        ctx.save_for_backward(vals, vecs)
        return vals, vecs

    @staticmethod
    def backward(ctx, g_vals, g_vecs):
        vals, vecs = ctx.saved_tensors
        return _eig_grad(vals, vecs, g_vals, g_vecs), None, None


class EigAttach(torch.autograd.Function):
    """Eigenpairs computed elsewhere (e.g. in a shared batch), attached to a's graph."""

    @staticmethod
    def forward(ctx, a, vals, vecs):
        ctx.save_for_backward(vals, vecs)
        return vals.clone(), vecs.clone()

    @staticmethod
    def backward(ctx, g_vals, g_vecs):
        vals, vecs = ctx.saved_tensors
        return _eig_grad(vals, vecs, g_vals, g_vecs), None, None


def _eig_grad(vals, vecs, g_vals, g_vecs):
    g_vals = torch.zeros_like(vals) if g_vals is None else g_vals
    g_vecs = torch.zeros_like(vecs) if g_vecs is None else g_vecs
    s = vals.unsqueeze(-2) - vals.unsqueeze(-1)            # s_ij = l_j - l_i
    f = torch.conj(s) / (torch.abs(s) ** 2 + BROADENING)
    f.diagonal(dim1=-2, dim2=-1).zero_()
    xh = vecs.mH
    inner = torch.conj(f) * (xh @ g_vecs)
    inner.diagonal(dim1=-2, dim2=-1).add_(g_vals)
    return torch.linalg.solve(xh, inner @ xh)


def batched_eig(a: torch.Tensor, warm: torch.Tensor | None = None, rows: torch.Tensor | None = None):
    return BatchedEig.apply(a, warm, rows)


def eig_attach(a: torch.Tensor, vals: torch.Tensor, vecs: torch.Tensor):
    return EigAttach.apply(a, vals, vecs)
