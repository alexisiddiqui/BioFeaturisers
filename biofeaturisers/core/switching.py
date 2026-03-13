"""Smooth switching functions and BV contact-count kernels."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .pairwise import dist_matrix_asymmetric


def sigmoid_switch(
    dist: Float[Array, "..."], r0: float, b: float
) -> Float[Array, "..."]:
    """Wan et al. sigmoid switch: ``sigmoid(b * (r0 - dist))``."""
    return jax.nn.sigmoid(b * (r0 - jnp.asarray(dist)))


def tanh_switch(dist: Float[Array, "..."], r0: float, k: float) -> Float[Array, "..."]:
    """Tanh switch equivalent to sigmoid with ``b = 2k``."""
    return 0.5 * (1.0 - jnp.tanh(k * (jnp.asarray(dist) - r0)))


def rational_switch(
    dist: Float[Array, "..."], r0: float, n: int = 6, m: int = 12
) -> Float[Array, "..."]:
    """PLUMED-style rational switch with removable singularity at ``r=r0``."""
    if r0 <= 0.0:
        raise ValueError("r0 must be > 0")
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be > 0")

    x = jnp.asarray(dist) / r0
    near_one = jnp.abs(x - 1.0) <= 1e-6
    x_safe = jnp.where(near_one, jnp.full_like(x, 0.5), x)
    ratio = (1.0 - x_safe**n) / (1.0 - x_safe**m)
    return jnp.where(near_one, jnp.asarray(float(n) / float(m), dtype=ratio.dtype), ratio)


def apply_switch_grid(
    dist_matrix: Float[Array, "n_probe n_env"],
    excl_mask: Float[Array, "n_probe n_env"],
    r0_grid: Float[Array, "n_r0"],
    b_grid: Float[Array, "n_b"],
) -> Float[Array, "n_r0 n_b n_probe"]:
    """Apply sigmoid switch over all ``(r0, b)`` combinations."""
    dist = jnp.asarray(dist_matrix)
    mask = jnp.asarray(excl_mask)
    r0_values = jnp.asarray(r0_grid)
    b_values = jnp.asarray(b_grid)

    chex.assert_rank(dist, 2)
    chex.assert_equal_shape((dist, mask))
    chex.assert_rank(r0_values, 1)
    chex.assert_rank(b_values, 1)

    switched = sigmoid_switch(
        dist[None, None, :, :],
        r0_values[:, None, None, None],
        b_values[None, :, None, None],
    )
    return jnp.sum(switched * mask[None, None, :, :], axis=-1)


def bv_contact_counts(
    coords: Float[Array, "n_atoms 3"],
    amide_N_idx: Int[Array, "n_probe"],
    amide_H_idx: Int[Array, "n_probe"],
    heavy_idx: Int[Array, "n_heavy"],
    backbone_O_idx: Int[Array, "n_backbone_o"],
    excl_mask_c: Float[Array, "n_probe n_heavy"],
    excl_mask_h: Float[Array, "n_probe n_backbone_o"],
    x_c: float = 6.5,
    x_h: float = 2.4,
    b: float = 10.0,
) -> tuple[Float[Array, "n_probe"], Float[Array, "n_probe"]]:
    """Compute Best-Vendruscolo heavy-atom and H-bond contact counts."""
    coords_arr = jnp.asarray(coords)
    chex.assert_rank(coords_arr, 2)
    if coords_arr.shape[1] != 3:
        raise ValueError(f"coords must have shape (n_atoms, 3), got {coords_arr.shape}")

    amide_n = coords_arr[jnp.asarray(amide_N_idx, dtype=jnp.int32)]
    amide_h = coords_arr[jnp.asarray(amide_H_idx, dtype=jnp.int32)]
    heavy = coords_arr[jnp.asarray(heavy_idx, dtype=jnp.int32)]
    backbone_o = coords_arr[jnp.asarray(backbone_O_idx, dtype=jnp.int32)]

    dist_c = dist_matrix_asymmetric(amide_n, heavy)
    dist_h = dist_matrix_asymmetric(amide_h, backbone_o)

    excl_c = jnp.asarray(excl_mask_c, dtype=dist_c.dtype)
    excl_h = jnp.asarray(excl_mask_h, dtype=dist_h.dtype)
    if excl_c.shape != dist_c.shape:
        raise ValueError(f"excl_mask_c shape {excl_c.shape} must match {dist_c.shape}")
    if excl_h.shape != dist_h.shape:
        raise ValueError(f"excl_mask_h shape {excl_h.shape} must match {dist_h.shape}")

    nc = jnp.sum(sigmoid_switch(dist_c, x_c, b) * excl_c, axis=-1)
    nh = jnp.sum(sigmoid_switch(dist_h, x_h, b) * excl_h, axis=-1)
    return nc, nh


__all__ = [
    "apply_switch_grid",
    "bv_contact_counts",
    "rational_switch",
    "sigmoid_switch",
    "tanh_switch",
]
