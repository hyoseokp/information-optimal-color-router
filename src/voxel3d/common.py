"""Helpers shared by the voxel3d drivers: ray rules, periodic filter, tanh projection, optional I_img context."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TRAIN_WAVELENGTHS_NM = (420.0, 450.0, 470.0, 510.0, 540.0, 570.0, 600.0, 635.0, 670.0)
RAY_RULES = {6: (1, 6), 8: (2, 4), 16: (2, 8)}


def gaussian_filter_periodic(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2.0 * sigma_px ** 2))
    k = k / k.sum()
    y = F.pad(x[None, None], (radius, radius, 0, 0), mode="circular")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, radius, radius), mode="circular")
    return F.conv2d(y, k.view(1, 1, -1, 1))[0, 0]


def project(x: torch.Tensor, beta: float, eta: float = 0.5) -> torch.Tensor:
    return (math.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (
        math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta)))


def load_context():
    """Matched imaging-information context (I_img). Returns None if it is not available in this checkout."""
    try:
        from cr_itd_v2.optimization.local_objectives import build_locked_matched_context
        return build_locked_matched_context()
    except Exception:
        return None


def i_img(tabs, context):
    """I_img of a [4, 9] spectral-efficiency table; NaN when no context is available (logging only)."""
    if context is None:
        return float("nan")
    from cr_itd_v2.optimization.local_objectives import evaluate_local_arm_objective
    return float(evaluate_local_arm_objective(tabs, context, "itd").objective)
