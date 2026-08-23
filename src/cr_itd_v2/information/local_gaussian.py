"""Local linear-Gaussian target information without spatial assumptions.

This module evaluates both the deterministic-target channel

``Y = R X + N`` and ``Z = T X``

for a zero-mean Gaussian scene vector ``X`` and independent zero-mean
Gaussian noise ``N``, and the more general joint-moment model specified by
``Sigma_XX``, ``Sigma_XZ``, and ``Sigma_ZZ``.  A direct channel-moment path
also accepts ``Sigma_YY_signal``, ``Sigma_YZ``, and ``Sigma_ZZ`` without
factoring a scene covariance, including a singular dense spectral latent.
The module deliberately knows nothing about spatial-frequency aliases,
fields, wavelengths, or the four-well CR phase order.

Array contract
--------------
The final two axes carry the linear-algebra axes and any leading axes are
independent batch coordinates::

    response                 R: B + [M, K]
    target_operator          T: B + [D, K]
    scene_covariance   Sigma_X: B + [K, K]
    noise_covariance   Sigma_N: B + [M, M]

The direct channel-moment path uses ``Sigma_YY_signal: B+[M,M]``,
``Sigma_YZ: B+[M,D]``, ``Sigma_ZZ: B+[D,D]``, and the same ``Sigma_N``.

For every input, ``B`` must either be absent (one matrix shared by all batch
entries) or exactly equal to the common batch shape.  Partial/suffix
broadcasting is rejected so a coincident dimension cannot silently acquire a
physical meaning.  The matrix axes use ordinary row/column order.  Spatial
origin, FFT convention, phase order, wavelength order, and physical units are
not applicable here; callers are responsible for supplying mutually
compatible units (in particular, ``Sigma_N`` must have squared ``Y`` units).

Real and complex tensors are supported.  Complex covariance matrices are
Hermitian and adjoints are used throughout.  The returned bit value follows
the CR manuscript's real-Gaussian convention

``0.5 * log2(det(Sigma_Z) / det(Sigma_{Z|Y}))``.

Complex tensors therefore provide Hermitian linear algebra for complex
operators (for example Fourier-domain intermediates); they do not switch the
normalization to the proper-complex Gaussian convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


_SUPPORTED_DTYPES = {
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
}


@dataclass(frozen=True)
class LocalTargetInformationResult:
    """Covariances and target information for a local Gaussian channel.

    All covariance fields have the common leading ``batch_shape``.  The
    ``information_bits`` tensor has exactly ``batch_shape`` (and is scalar for
    an unbatched channel).
    """

    information_bits: torch.Tensor
    posterior_target_covariance: torch.Tensor
    target_covariance: torch.Tensor
    measurement_covariance: torch.Tensor
    measurement_signal_covariance: torch.Tensor
    measurement_target_cross_covariance: torch.Tensor

    @property
    def bits(self) -> torch.Tensor:
        """Short, read-only alias for :attr:`information_bits`."""

        return self.information_bits

    @property
    def sigma_z_given_y(self) -> torch.Tensor:
        """Notation-oriented alias for the posterior target covariance."""

        return self.posterior_target_covariance

    @property
    def sigma_z(self) -> torch.Tensor:
        """Notation-oriented alias for the prior target covariance."""

        return self.target_covariance

    @property
    def sigma_y(self) -> torch.Tensor:
        """Notation-oriented alias for the total measurement covariance."""

        return self.measurement_covariance

    @property
    def sigma_yz(self) -> torch.Tensor:
        """Notation-oriented alias for ``Cov(Y, Z)``."""

        return self.measurement_target_cross_covariance


@dataclass(frozen=True)
class JointLocalTargetInformationResult:
    """Target information from explicitly supplied joint scene/target moments.

    Unlike :class:`LocalTargetInformationResult`, this result does not require
    the target to be a deterministic linear transform of the measurement
    latent.  ``scene_target_cross_covariance`` is ``Cov(X,Z)``.  Consequently,
    ``target_given_scene_residual_covariance`` retains target-relevant spectral
    nuisance that is not determined by the K-component measurement latent.
    """

    information_bits: torch.Tensor
    posterior_target_covariance: torch.Tensor
    recovered_target_covariance: torch.Tensor
    target_covariance: torch.Tensor
    measurement_covariance: torch.Tensor
    measurement_signal_covariance: torch.Tensor
    measurement_target_cross_covariance: torch.Tensor
    posterior_scene_covariance: torch.Tensor
    target_given_scene_residual_covariance: torch.Tensor

    @property
    def bits(self) -> torch.Tensor:
        """Short alias for :attr:`information_bits`."""

        return self.information_bits

    @property
    def sigma_z_given_y(self) -> torch.Tensor:
        """Notation-oriented alias for the posterior target covariance."""

        return self.posterior_target_covariance

    @property
    def sigma_z(self) -> torch.Tensor:
        """Notation-oriented alias for the prior target covariance."""

        return self.target_covariance

    @property
    def sigma_y(self) -> torch.Tensor:
        """Notation-oriented alias for the total measurement covariance."""

        return self.measurement_covariance

    @property
    def sigma_yz(self) -> torch.Tensor:
        """Notation-oriented alias for ``Cov(Y,Z)``."""

        return self.measurement_target_cross_covariance


@dataclass(frozen=True)
class ChannelMomentTargetInformationResult:
    """Target information from direct measurement/target channel moments.

    This contract does not expose or factor a scene-latent covariance.  It is
    therefore suitable when a dense spectral covariance is only positive
    semidefinite.  ``measurement_signal_covariance`` is the noise-free
    measured-channel covariance and ``measurement_target_cross_covariance``
    is ``Cov(Y_signal,Z)``.
    """

    information_bits: torch.Tensor
    posterior_target_covariance: torch.Tensor
    recovered_target_covariance: torch.Tensor
    target_covariance: torch.Tensor
    measurement_covariance: torch.Tensor
    measurement_signal_covariance: torch.Tensor
    measurement_target_cross_covariance: torch.Tensor

    @property
    def bits(self) -> torch.Tensor:
        """Short alias for :attr:`information_bits`."""

        return self.information_bits

    @property
    def sigma_z_given_y(self) -> torch.Tensor:
        """Notation-oriented alias for the posterior target covariance."""

        return self.posterior_target_covariance

    @property
    def sigma_z(self) -> torch.Tensor:
        """Notation-oriented alias for the prior target covariance."""

        return self.target_covariance

    @property
    def sigma_y(self) -> torch.Tensor:
        """Notation-oriented alias for the total measurement covariance."""

        return self.measurement_covariance

    @property
    def sigma_yz(self) -> torch.Tensor:
        """Notation-oriented alias for ``Cov(Y,Z)``."""

        return self.measurement_target_cross_covariance


def _adjoint(value: torch.Tensor) -> torch.Tensor:
    return value.transpose(-2, -1).conj()


def _hermitian(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + _adjoint(value))


def _as_numeric_tensor(value: object, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"{name} must use float32, float64, complex64, or complex128; "
            f"got {tensor.dtype}"
        )
    return tensor


def _common_device(values: Sequence[tuple[str, torch.Tensor]]) -> torch.device:
    device = values[0][1].device
    for name, value in values[1:]:
        if value.device != device:
            raise ValueError(
                f"all inputs must be on one device; {name} is on "
                f"{value.device}, expected {device}"
            )
    return device


def _common_dtype(values: Sequence[tuple[str, torch.Tensor]]) -> torch.dtype:
    dtype = values[0][1].dtype
    for _, value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported promoted dtype {dtype}")
    return dtype


def _matrix_batch_shape(value: torch.Tensor, name: str) -> tuple[int, ...]:
    if value.ndim < 2:
        raise ValueError(f"{name} must have at least two matrix axes")
    return tuple(map(int, value.shape[:-2]))


def _resolve_batch_shape(
    values: Sequence[tuple[str, torch.Tensor]],
) -> tuple[int, ...]:
    nonempty = {
        _matrix_batch_shape(value, name)
        for name, value in values
        if _matrix_batch_shape(value, name)
    }
    if len(nonempty) > 1:
        rendered = ", ".join(
            f"{name}={_matrix_batch_shape(value, name)}"
            for name, value in values
        )
        raise ValueError(
            "leading batch axes must be absent or exactly equal; got " + rendered
        )
    return next(iter(nonempty), ())


def _expand_common_matrix(
    value: torch.Tensor,
    *,
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    if not batch_shape or value.ndim > 2:
        return value
    rows, columns = map(int, value.shape)
    return value.reshape(
        *((1,) * len(batch_shape)), rows, columns
    ).expand(*batch_shape, rows, columns)


def _validate_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_hermitian(value: torch.Tensor, name: str) -> torch.Tensor:
    difference = value - _adjoint(value)
    real_dtype = value.real.dtype if value.is_complex() else value.dtype
    tolerance = 256.0 * torch.finfo(real_dtype).eps
    scale = torch.maximum(
        value.detach().abs().amax(),
        torch.ones((), device=value.device, dtype=real_dtype),
    )
    if bool((difference.detach().abs().amax() > tolerance * scale).item()):
        raise ValueError(f"{name} must be Hermitian")
    return _hermitian(value)


def _checked_cholesky(value: torch.Tensor, name: str) -> torch.Tensor:
    factor, info = torch.linalg.cholesky_ex(value)
    failed = info.detach() != 0
    if bool(failed.any()):
        locations = torch.nonzero(failed, as_tuple=False)
        if locations.numel() == 0:
            location = ""
        elif failed.ndim == 0:
            location = ""
        else:
            location = f" at batch index {tuple(map(int, locations[0].tolist()))}"
        raise ValueError(f"{name} must be positive definite{location}")
    return factor


def _validate_hermitian_psd(
    value: torch.Tensor,
    name: str,
    *,
    reference_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate a Hermitian positive-semidefinite matrix without regularizing it."""

    hermitian = _validate_hermitian(value, name)
    eigenvalues = torch.linalg.eigvalsh(hermitian)
    real_dtype = eigenvalues.dtype
    dimension = int(hermitian.shape[-1])
    scale = hermitian.detach().abs().amax(dim=(-2, -1)).clamp_min(
        torch.finfo(real_dtype).tiny
    )
    if reference_scale is not None:
        reference = torch.as_tensor(
            reference_scale, dtype=real_dtype, device=hermitian.device
        )
        if tuple(reference.shape) != tuple(scale.shape):
            raise ValueError(f"{name} PSD reference scale has the wrong batch shape")
        if not bool(torch.isfinite(reference).all()) or bool((reference < 0).any()):
            raise ValueError(f"{name} PSD reference scale must be finite/non-negative")
        scale = torch.maximum(scale, reference)
    tolerance = (
        2048.0 * max(dimension, 1) * torch.finfo(real_dtype).eps * scale
    )
    if bool((eigenvalues.detach() < -tolerance.unsqueeze(-1)).any()):
        locations = torch.nonzero(
            eigenvalues.detach() < -tolerance.unsqueeze(-1), as_tuple=False
        )
        location = ""
        if eigenvalues.ndim > 1 and locations.numel():
            location = f" at batch index {tuple(map(int, locations[0, :-1].tolist()))}"
        raise ValueError(f"{name} must be positive semidefinite{location}")
    return hermitian


def _logdet_from_cholesky(factor: torch.Tensor) -> torch.Tensor:
    diagonal = torch.diagonal(factor, dim1=-2, dim2=-1).real
    return 2.0 * torch.log(diagonal).sum(dim=-1)


def evaluate_local_target_information(
    response: torch.Tensor,
    scene_covariance: torch.Tensor,
    target_operator: torch.Tensor,
    noise_covariance: torch.Tensor,
) -> LocalTargetInformationResult:
    """Evaluate local ``I(Y; Z)`` and ``Cov(Z | Y)`` in bits.

    Args:
        response: ``R`` with shape ``B + [M,K]``.
        scene_covariance: Hermitian positive-definite ``Sigma_X`` with shape
            ``B + [K,K]``.
        target_operator: ``T`` with shape ``B + [D,K]``.
        noise_covariance: Hermitian positive-definite ``Sigma_N`` with shape
            ``B + [M,M]``.  Noise is assumed independent of ``X``.

    Returns:
        A :class:`LocalTargetInformationResult` with bit shape ``B``, target
        covariance shape ``B + [D,D]``, measurement covariance shape
        ``B + [M,M]``, and cross-covariance shape ``B + [M,D]``.

    Raises:
        ValueError: If a shape/dtype/device contract is violated, an input is
        non-finite or non-Hermitian, or a required covariance is not positive
        definite.  No implicit jitter or eigenvalue clipping is applied.
    """

    named = [
        ("response", _as_numeric_tensor(response, "response")),
        (
            "target_operator",
            _as_numeric_tensor(target_operator, "target_operator"),
        ),
        (
            "scene_covariance",
            _as_numeric_tensor(scene_covariance, "scene_covariance"),
        ),
        (
            "noise_covariance",
            _as_numeric_tensor(noise_covariance, "noise_covariance"),
        ),
    ]
    device = _common_device(named)
    dtype = _common_dtype(named)
    tensors = {
        name: value.to(device=device, dtype=dtype) for name, value in named
    }
    response_tensor = tensors["response"]
    target_tensor = tensors["target_operator"]
    scene = tensors["scene_covariance"]
    noise = tensors["noise_covariance"]

    batch_shape = _resolve_batch_shape(
        [
            ("response", response_tensor),
            ("target_operator", target_tensor),
            ("scene_covariance", scene),
            ("noise_covariance", noise),
        ]
    )
    response_tensor = _expand_common_matrix(
        response_tensor, batch_shape=batch_shape
    )
    target_tensor = _expand_common_matrix(target_tensor, batch_shape=batch_shape)
    scene = _expand_common_matrix(scene, batch_shape=batch_shape)
    noise = _expand_common_matrix(noise, batch_shape=batch_shape)

    measurement_rows, latent_channels = map(int, response_tensor.shape[-2:])
    target_rows, target_latent = map(int, target_tensor.shape[-2:])
    if measurement_rows <= 0 or latent_channels <= 0 or target_rows <= 0:
        raise ValueError("M, K, and D must all be positive")
    if target_latent != latent_channels:
        raise ValueError(
            "response and target_operator must share their final latent K axis"
        )
    if tuple(scene.shape[-2:]) != (latent_channels, latent_channels):
        raise ValueError("scene_covariance must have shape B + [K,K]")
    if tuple(noise.shape[-2:]) != (measurement_rows, measurement_rows):
        raise ValueError("noise_covariance must have shape B + [M,M]")

    for name, value in (
        ("response", response_tensor),
        ("target_operator", target_tensor),
        ("scene_covariance", scene),
        ("noise_covariance", noise),
    ):
        _validate_finite(value, name)

    scene = _validate_hermitian(scene, "scene_covariance")
    noise = _validate_hermitian(noise, "noise_covariance")
    chol_scene = _checked_cholesky(scene, "scene_covariance")
    chol_noise = _checked_cholesky(noise, "noise_covariance")

    response_scene = response_tensor @ scene
    target_scene = target_tensor @ scene
    sigma_y_signal = _hermitian(response_scene @ _adjoint(response_tensor))
    sigma_y = _hermitian(sigma_y_signal + noise)
    sigma_z = _hermitian(target_scene @ _adjoint(target_tensor))
    sigma_yz = response_scene @ _adjoint(target_tensor)

    _checked_cholesky(sigma_y, "measurement_covariance")
    chol_z = _checked_cholesky(sigma_z, "target_covariance")

    # Build the posterior through a whitened information matrix rather than
    # subtracting two nearly equal covariance matrices.  With
    # X = chol_scene @ U and whitened noise, Cov(U | Y) is
    # (I + C^H C)^-1, where C = chol_noise^-1 R chol_scene.
    whitened_response = torch.linalg.solve_triangular(
        chol_noise,
        response_tensor @ chol_scene,
        upper=False,
    )
    identity = torch.eye(
        latent_channels, dtype=dtype, device=device
    ).reshape(*((1,) * len(batch_shape)), latent_channels, latent_channels)
    whitened_information = _hermitian(
        identity + _adjoint(whitened_response) @ whitened_response
    )
    chol_whitened_information = _checked_cholesky(
        whitened_information, "whitened_scene_information"
    )
    whitened_target = target_tensor @ chol_scene
    solved_target = torch.cholesky_solve(
        _adjoint(whitened_target), chol_whitened_information
    )
    sigma_z_given_y = _hermitian(whitened_target @ solved_target)
    chol_z_given_y = _checked_cholesky(
        sigma_z_given_y, "posterior_target_covariance"
    )

    information_nats = 0.5 * (
        _logdet_from_cholesky(chol_z)
        - _logdet_from_cholesky(chol_z_given_y)
    )
    information_bits = information_nats / math.log(2.0)

    return LocalTargetInformationResult(
        information_bits=information_bits,
        posterior_target_covariance=sigma_z_given_y,
        target_covariance=sigma_z,
        measurement_covariance=sigma_y,
        measurement_signal_covariance=sigma_y_signal,
        measurement_target_cross_covariance=sigma_yz,
    )


def evaluate_joint_target_information(
    response: torch.Tensor,
    scene_covariance: torch.Tensor,
    scene_target_cross_covariance: torch.Tensor,
    target_covariance: torch.Tensor,
    noise_covariance: torch.Tensor,
) -> JointLocalTargetInformationResult:
    """Evaluate ``I(Y;Z)`` from explicit joint moments of ``X`` and ``Z``.

    The channel is ``Y = R X + N`` with independent Gaussian ``N``.  Inputs
    have shapes ``R: B+[M,K]``, ``Sigma_XX: B+[K,K]``,
    ``Sigma_XZ: B+[K,D]``, ``Sigma_ZZ: B+[D,D]``, and
    ``Sigma_N: B+[M,M]``.  Leading batch axes must be absent or exactly equal.

    A valid joint covariance is required.  The target may contain an
    irreducible component not determined by ``X``; equivalently,

    ``Z = C X + epsilon``

    where ``C = Sigma_ZX Sigma_XX^-1`` and ``epsilon`` is independent of
    ``X`` in the Gaussian model.  The posterior is formed as

    ``Sigma_(Z|Y) = Sigma_epsilon + C Sigma_(X|Y) C^H``

    so a small target-nuisance floor is retained without subtracting two nearly
    equal target covariances.  No jitter or covariance eigenvalue clipping is
    applied; malformed or materially indefinite moments fail closed.
    """

    named = [
        ("response", _as_numeric_tensor(response, "response")),
        (
            "scene_covariance",
            _as_numeric_tensor(scene_covariance, "scene_covariance"),
        ),
        (
            "scene_target_cross_covariance",
            _as_numeric_tensor(
                scene_target_cross_covariance,
                "scene_target_cross_covariance",
            ),
        ),
        (
            "target_covariance",
            _as_numeric_tensor(target_covariance, "target_covariance"),
        ),
        (
            "noise_covariance",
            _as_numeric_tensor(noise_covariance, "noise_covariance"),
        ),
    ]
    device = _common_device(named)
    dtype = _common_dtype(named)
    tensors = {
        name: value.to(device=device, dtype=dtype) for name, value in named
    }
    response_tensor = tensors["response"]
    scene = tensors["scene_covariance"]
    scene_target = tensors["scene_target_cross_covariance"]
    target = tensors["target_covariance"]
    noise = tensors["noise_covariance"]

    batch_shape = _resolve_batch_shape(
        [
            ("response", response_tensor),
            ("scene_covariance", scene),
            ("scene_target_cross_covariance", scene_target),
            ("target_covariance", target),
            ("noise_covariance", noise),
        ]
    )
    response_tensor = _expand_common_matrix(
        response_tensor, batch_shape=batch_shape
    )
    scene = _expand_common_matrix(scene, batch_shape=batch_shape)
    scene_target = _expand_common_matrix(
        scene_target, batch_shape=batch_shape
    )
    target = _expand_common_matrix(target, batch_shape=batch_shape)
    noise = _expand_common_matrix(noise, batch_shape=batch_shape)

    measurement_rows, latent_channels = map(int, response_tensor.shape[-2:])
    cross_latent, target_rows = map(int, scene_target.shape[-2:])
    if measurement_rows <= 0 or latent_channels <= 0 or target_rows <= 0:
        raise ValueError("M, K, and D must all be positive")
    if cross_latent != latent_channels:
        raise ValueError(
            "scene_target_cross_covariance must share response's latent K axis"
        )
    if tuple(scene.shape[-2:]) != (latent_channels, latent_channels):
        raise ValueError("scene_covariance must have shape B + [K,K]")
    if tuple(target.shape[-2:]) != (target_rows, target_rows):
        raise ValueError("target_covariance must have shape B + [D,D]")
    if tuple(noise.shape[-2:]) != (measurement_rows, measurement_rows):
        raise ValueError("noise_covariance must have shape B + [M,M]")

    for name, value in (
        ("response", response_tensor),
        ("scene_covariance", scene),
        ("scene_target_cross_covariance", scene_target),
        ("target_covariance", target),
        ("noise_covariance", noise),
    ):
        _validate_finite(value, name)

    scene = _validate_hermitian(scene, "scene_covariance")
    target = _validate_hermitian(target, "target_covariance")
    noise = _validate_hermitian(noise, "noise_covariance")
    chol_scene = _checked_cholesky(scene, "scene_covariance")
    chol_target = _checked_cholesky(target, "target_covariance")
    chol_noise = _checked_cholesky(noise, "noise_covariance")

    joint_top = torch.cat((scene, scene_target), dim=-1)
    joint_bottom = torch.cat((_adjoint(scene_target), target), dim=-1)
    joint = torch.cat((joint_top, joint_bottom), dim=-2)
    _validate_hermitian_psd(joint, "joint_scene_target_covariance")

    # Regression of Z on X and its irreducible conditional covariance.
    precision_times_cross = torch.cholesky_solve(scene_target, chol_scene)
    regression = _adjoint(precision_times_cross)
    target_given_scene = _hermitian(
        target - _adjoint(scene_target) @ precision_times_cross
    )
    target_given_scene = _validate_hermitian_psd(
        target_given_scene,
        "target_given_scene_residual_covariance",
        reference_scale=target.detach().abs().amax(dim=(-2, -1)),
    )

    response_scene = response_tensor @ scene
    sigma_y_signal = _hermitian(response_scene @ _adjoint(response_tensor))
    sigma_y = _hermitian(sigma_y_signal + noise)
    chol_y = _checked_cholesky(sigma_y, "measurement_covariance")
    sigma_yz = response_tensor @ scene_target
    recovered = _hermitian(
        _adjoint(sigma_yz) @ torch.cholesky_solve(sigma_yz, chol_y)
    )
    recovered = _validate_hermitian_psd(
        recovered, "recovered_target_covariance"
    )

    # Stable scene posterior in whitened latent coordinates.
    whitened_response = torch.linalg.solve_triangular(
        chol_noise,
        response_tensor @ chol_scene,
        upper=False,
    )
    identity = torch.eye(
        latent_channels, dtype=dtype, device=device
    ).reshape(*((1,) * len(batch_shape)), latent_channels, latent_channels)
    whitened_information = _hermitian(
        identity + _adjoint(whitened_response) @ whitened_response
    )
    chol_whitened_information = _checked_cholesky(
        whitened_information, "whitened_scene_information"
    )
    posterior_whitened_scene = torch.cholesky_solve(
        identity, chol_whitened_information
    )
    posterior_scene = _hermitian(
        chol_scene @ posterior_whitened_scene @ _adjoint(chol_scene)
    )
    posterior_target = _hermitian(
        target_given_scene
        + regression @ posterior_scene @ _adjoint(regression)
    )
    chol_posterior_target = _checked_cholesky(
        posterior_target, "posterior_target_covariance"
    )

    # The stable posterior and direct recovered covariance must describe the
    # same Schur complement.  Fail rather than silently accepting inconsistent
    # user moments or unstable arithmetic.
    decomposition_error = (target - recovered - posterior_target).detach().abs().amax()
    real_dtype = posterior_target.real.dtype if posterior_target.is_complex() else posterior_target.dtype
    decomposition_scale = torch.stack(
        (
            target.detach().abs().amax(),
            posterior_target.detach().abs().amax(),
            torch.ones((), dtype=real_dtype, device=device),
        )
    ).amax()
    if bool(
        decomposition_error
        > 4096.0 * torch.finfo(real_dtype).eps * decomposition_scale
    ):
        raise ValueError(
            "joint target posterior/recovered covariance decomposition is inconsistent"
        )

    information_nats = 0.5 * (
        _logdet_from_cholesky(chol_target)
        - _logdet_from_cholesky(chol_posterior_target)
    )
    information_bits = information_nats / math.log(2.0)
    return JointLocalTargetInformationResult(
        information_bits=information_bits,
        posterior_target_covariance=posterior_target,
        recovered_target_covariance=recovered,
        target_covariance=target,
        measurement_covariance=sigma_y,
        measurement_signal_covariance=sigma_y_signal,
        measurement_target_cross_covariance=sigma_yz,
        posterior_scene_covariance=posterior_scene,
        target_given_scene_residual_covariance=target_given_scene,
    )


def evaluate_channel_moment_target_information(
    measurement_signal_covariance: torch.Tensor,
    measurement_target_cross_covariance: torch.Tensor,
    target_covariance: torch.Tensor,
    noise_covariance: torch.Tensor,
) -> ChannelMomentTargetInformationResult:
    """Evaluate ``I(Y;Z)`` directly from channel and target moments.

    The supplied moments are ``Sigma_SS=Cov(Y_signal,Y_signal)``,
    ``Sigma_SZ=Cov(Y_signal,Z)``, ``Sigma_ZZ``, and independent measurement
    noise ``Sigma_N``.  ``Sigma_SS`` and the joint signal/target covariance may
    be singular; neither is Cholesky-factored.  ``Sigma_ZZ``, ``Sigma_N``, the
    total measurement covariance, and the posterior target covariance must be
    positive definite.  No jitter or eigenvalue clipping is applied.
    """

    named = [
        (
            "measurement_signal_covariance",
            _as_numeric_tensor(
                measurement_signal_covariance,
                "measurement_signal_covariance",
            ),
        ),
        (
            "measurement_target_cross_covariance",
            _as_numeric_tensor(
                measurement_target_cross_covariance,
                "measurement_target_cross_covariance",
            ),
        ),
        (
            "target_covariance",
            _as_numeric_tensor(target_covariance, "target_covariance"),
        ),
        (
            "noise_covariance",
            _as_numeric_tensor(noise_covariance, "noise_covariance"),
        ),
    ]
    device = _common_device(named)
    dtype = _common_dtype(named)
    tensors = {
        name: value.to(device=device, dtype=dtype) for name, value in named
    }
    signal = tensors["measurement_signal_covariance"]
    cross = tensors["measurement_target_cross_covariance"]
    target = tensors["target_covariance"]
    noise = tensors["noise_covariance"]
    batch_shape = _resolve_batch_shape(
        [
            ("measurement_signal_covariance", signal),
            ("measurement_target_cross_covariance", cross),
            ("target_covariance", target),
            ("noise_covariance", noise),
        ]
    )
    signal = _expand_common_matrix(signal, batch_shape=batch_shape)
    cross = _expand_common_matrix(cross, batch_shape=batch_shape)
    target = _expand_common_matrix(target, batch_shape=batch_shape)
    noise = _expand_common_matrix(noise, batch_shape=batch_shape)

    measurement_rows, signal_columns = map(int, signal.shape[-2:])
    cross_measurements, target_rows = map(int, cross.shape[-2:])
    if measurement_rows <= 0 or target_rows <= 0:
        raise ValueError("M and D must both be positive")
    if signal_columns != measurement_rows:
        raise ValueError(
            "measurement_signal_covariance must have shape B + [M,M]"
        )
    if cross_measurements != measurement_rows:
        raise ValueError(
            "measurement_target_cross_covariance must have shape B + [M,D]"
        )
    if tuple(target.shape[-2:]) != (target_rows, target_rows):
        raise ValueError("target_covariance must have shape B + [D,D]")
    if tuple(noise.shape[-2:]) != (measurement_rows, measurement_rows):
        raise ValueError("noise_covariance must have shape B + [M,M]")
    for name, value in (
        ("measurement_signal_covariance", signal),
        ("measurement_target_cross_covariance", cross),
        ("target_covariance", target),
        ("noise_covariance", noise),
    ):
        _validate_finite(value, name)

    signal = _validate_hermitian_psd(
        signal, "measurement_signal_covariance"
    )
    target = _validate_hermitian(target, "target_covariance")
    noise = _validate_hermitian(noise, "noise_covariance")
    chol_target = _checked_cholesky(target, "target_covariance")
    _checked_cholesky(noise, "noise_covariance")
    joint = torch.cat(
        (
            torch.cat((signal, cross), dim=-1),
            torch.cat((_adjoint(cross), target), dim=-1),
        ),
        dim=-2,
    )
    _validate_hermitian_psd(
        joint, "joint_measurement_signal_target_covariance"
    )

    measurement = _hermitian(signal + noise)
    chol_measurement = _checked_cholesky(
        measurement, "measurement_covariance"
    )
    recovered = _hermitian(
        _adjoint(cross) @ torch.cholesky_solve(cross, chol_measurement)
    )
    recovered = _validate_hermitian_psd(
        recovered, "recovered_target_covariance"
    )
    posterior = _hermitian(target - recovered)
    chol_posterior = _checked_cholesky(
        posterior, "posterior_target_covariance"
    )
    information_nats = 0.5 * (
        _logdet_from_cholesky(chol_target)
        - _logdet_from_cholesky(chol_posterior)
    )
    information_bits = information_nats / math.log(2.0)
    return ChannelMomentTargetInformationResult(
        information_bits=information_bits,
        posterior_target_covariance=posterior,
        recovered_target_covariance=recovered,
        target_covariance=target,
        measurement_covariance=measurement,
        measurement_signal_covariance=signal,
        measurement_target_cross_covariance=cross,
    )


def local_target_information(
    response: torch.Tensor,
    scene_covariance: torch.Tensor,
    target_operator: torch.Tensor,
    noise_covariance: torch.Tensor,
) -> LocalTargetInformationResult:
    """Alias for :func:`evaluate_local_target_information`."""

    return evaluate_local_target_information(
        response, scene_covariance, target_operator, noise_covariance
    )


def joint_target_information(
    response: torch.Tensor,
    scene_covariance: torch.Tensor,
    scene_target_cross_covariance: torch.Tensor,
    target_covariance: torch.Tensor,
    noise_covariance: torch.Tensor,
) -> JointLocalTargetInformationResult:
    """Alias for :func:`evaluate_joint_target_information`."""

    return evaluate_joint_target_information(
        response,
        scene_covariance,
        scene_target_cross_covariance,
        target_covariance,
        noise_covariance,
    )


def channel_moment_target_information(
    measurement_signal_covariance: torch.Tensor,
    measurement_target_cross_covariance: torch.Tensor,
    target_covariance: torch.Tensor,
    noise_covariance: torch.Tensor,
) -> ChannelMomentTargetInformationResult:
    """Alias for :func:`evaluate_channel_moment_target_information`."""

    return evaluate_channel_moment_target_information(
        measurement_signal_covariance,
        measurement_target_cross_covariance,
        target_covariance,
        noise_covariance,
    )


__all__ = [
    "ChannelMomentTargetInformationResult",
    "JointLocalTargetInformationResult",
    "LocalTargetInformationResult",
    "channel_moment_target_information",
    "evaluate_channel_moment_target_information",
    "evaluate_joint_target_information",
    "evaluate_local_target_information",
    "joint_target_information",
    "local_target_information",
]
