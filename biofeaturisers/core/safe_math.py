from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

_SINC_TAYLOR_THRESHOLD = 1e-8


def safe_sqrt(x: Float[Array, "..."], eps: float = 1e-10) -> Float[Array, "..."]:
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive")
    return jnp.sqrt(x + eps)


def safe_sqrt_sym(dist_sq: Float[Array, "B B"]) -> Float[Array, "B B"]:
    chex.assert_rank(dist_sq, 2)
    if dist_sq.shape[0] != dist_sq.shape[1]:
        raise ValueError("dist_sq must be square")

    is_zero = dist_sq <= 0.0
    safe_sq = jnp.where(is_zero, 1.0, dist_sq)
    return jnp.where(is_zero, 0.0, jnp.sqrt(safe_sq))


@jax.custom_vjp
def safe_sinc(qr: Float[Array, "..."]) -> Float[Array, "..."]:
    use_standard_branch = jnp.abs(qr) > _SINC_TAYLOR_THRESHOLD
    safe_qr = jnp.where(use_standard_branch, qr, 1.0)
    taylor = 1.0 - qr**2 / 6.0 + qr**4 / 120.0
    return jnp.where(use_standard_branch, jnp.sin(safe_qr) / safe_qr, taylor)


def _safe_sinc_fwd(qr: Float[Array, "..."]) -> tuple[Float[Array, "..."], tuple[Float[Array, "..."], Float[Array, "..."]]]:
    y = safe_sinc(qr)
    return y, (qr, y)


def _safe_sinc_bwd(
    res: tuple[Float[Array, "..."], Float[Array, "..."]],
    g: Float[Array, "..."],
) -> tuple[Float[Array, "..."]]:
    qr, y = res
    use_standard_branch = jnp.abs(qr) > _SINC_TAYLOR_THRESHOLD
    safe_qr = jnp.where(use_standard_branch, qr, 1.0)

    dsinc = jnp.where(
        use_standard_branch,
        (jnp.cos(safe_qr) - y) / safe_qr,
        -qr / 3.0 + qr**3 / 30.0,
    )
    return (g * dsinc,)


safe_sinc.defvjp(_safe_sinc_fwd, _safe_sinc_bwd)


def safe_mask(
    mask: Array,
    fn: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    operand: Float[Array, "..."],
    placeholder: float = 0.0,
    safe_val: float = 0.5,
) -> Float[Array, "..."]:
    mask_array = jnp.asarray(mask, dtype=bool)
    operand_array = jnp.asarray(operand)
    chex.assert_equal_shape([mask_array, operand_array])

    safe_operand = jnp.where(mask_array, operand_array, jnp.asarray(safe_val, dtype=operand_array.dtype))
    transformed = fn(safe_operand)
    return jnp.where(mask_array, transformed, jnp.asarray(placeholder, dtype=operand_array.dtype))


def diagonal_self_pairs(ff: Float[Array, "N Q"]) -> Float[Array, "Q"]:
    chex.assert_rank(ff, 2)
    return jnp.sum(ff**2, axis=0)

