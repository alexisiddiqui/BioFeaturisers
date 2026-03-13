"""Numerically safe math primitives for pairwise kernels."""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

_SINC_TAYLOR_SWITCH = 1e-8


def safe_sqrt(x: Float[Array, "..."], eps: float = 1e-10) -> Float[Array, "..."]:
    """Return ``sqrt(x + eps)`` with finite gradient everywhere."""
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    return jnp.sqrt(jnp.asarray(x) + eps)


def safe_sqrt_sym(dist_sq: Float[Array, "n_i n_j"]) -> Float[Array, "n_i n_j"]:
    """Safe sqrt for blocks that may contain exact zeros."""
    dist_sq_arr = jnp.asarray(dist_sq)
    chex.assert_rank(dist_sq_arr, 2)

    is_zero = dist_sq_arr <= 0.0
    safe_sq = jnp.where(is_zero, jnp.ones_like(dist_sq_arr), dist_sq_arr)
    return jnp.where(is_zero, jnp.zeros_like(dist_sq_arr), jnp.sqrt(safe_sq))


def safe_mask(
    mask: Bool[Array, "..."] | Float[Array, "..."] | Array,
    fn: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    operand: Float[Array, "..."],
    placeholder: float = 0.0,
    safe_val: float = 0.5,
) -> Float[Array, "..."]:
    """Apply ``fn`` on masked values only, while keeping gradients finite."""
    operand_arr = jnp.asarray(operand)
    mask_arr = jnp.asarray(mask, dtype=jnp.bool_)
    chex.assert_equal_shape((mask_arr, operand_arr))

    safe_operand = jnp.where(mask_arr, operand_arr, jnp.asarray(safe_val, dtype=operand_arr.dtype))
    masked_result = fn(safe_operand)
    return jnp.where(
        mask_arr,
        masked_result,
        jnp.asarray(placeholder, dtype=masked_result.dtype),
    )


def diagonal_self_pairs(ff: Float[Array, "n q"]) -> Float[Array, "q"]:
    """Return diagonal self-term contribution ``sum_i ff[i, q]^2``."""
    ff_arr = jnp.asarray(ff)
    chex.assert_rank(ff_arr, 2)
    return jnp.sum(ff_arr * ff_arr, axis=0)


@jax.custom_vjp
def safe_sinc(qr: Float[Array, "..."]) -> Float[Array, "..."]:
    """Stable ``sin(x)/x`` with a Taylor branch around zero."""
    qr_arr = jnp.asarray(qr)
    large = jnp.abs(qr_arr) > _SINC_TAYLOR_SWITCH
    safe_qr = jnp.where(large, qr_arr, jnp.ones_like(qr_arr))
    normal = jnp.sin(safe_qr) / safe_qr
    taylor = 1.0 - qr_arr**2 / 6.0 + qr_arr**4 / 120.0
    return jnp.where(large, normal, taylor)


def _safe_sinc_fwd(qr: Float[Array, "..."]) -> tuple[Float[Array, "..."], tuple[Array, Array]]:
    y = safe_sinc(qr)
    return y, (jnp.asarray(qr), jnp.asarray(y))


def _safe_sinc_bwd(
    res: tuple[Array, Array], g: Float[Array, "..."]
) -> tuple[Float[Array, "..."]]:
    qr, y = res
    qr_arr = jnp.asarray(qr)
    large = jnp.abs(qr_arr) > _SINC_TAYLOR_SWITCH
    safe_qr = jnp.where(large, qr_arr, jnp.ones_like(qr_arr))
    dsinc = jnp.where(
        large,
        (jnp.cos(safe_qr) - y) / safe_qr,
        -qr_arr / 3.0 + qr_arr**3 / 30.0,
    )
    return (jnp.asarray(g) * dsinc,)


safe_sinc.defvjp(_safe_sinc_fwd, _safe_sinc_bwd)


__all__ = [
    "diagonal_self_pairs",
    "safe_mask",
    "safe_sinc",
    "safe_sqrt",
    "safe_sqrt_sym",
]
