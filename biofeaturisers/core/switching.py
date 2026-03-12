from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .pairwise import dist_matrix_asymmetric


def sigmoid_switch(dist: Float[Array, "..."], r0: float, b: float) -> Float[Array, "..."]:
    return jax.nn.sigmoid(b * (r0 - dist))


def tanh_switch(dist: Float[Array, "..."], r0: float, k: float) -> Float[Array, "..."]:
    return 0.5 * (1.0 - jnp.tanh(k * (dist - r0)))


def rational_switch(
    dist: Float[Array, "..."],
    r0: float,
    n: int = 6,
    m: int = 12,
) -> Float[Array, "..."]:
    if r0 <= 0.0:
        raise ValueError("r0 must be strictly positive")
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")

    x = dist / r0
    near_one = jnp.abs(x - 1.0) < 1e-6
    x_safe = jnp.where(near_one, 0.5, x)
    switched = (1.0 - x_safe**n) / (1.0 - x_safe**m)
    return jnp.where(near_one, float(n) / float(m), switched)


def apply_switch_grid(
    dist_matrix: Float[Array, "N_p N_e"],
    excl_mask: Float[Array, "N_p N_e"] | Array,
    r0_grid: Float[Array, "K"],
    b_grid: Float[Array, "L"],
) -> Float[Array, "K L N_p"]:
    chex.assert_rank([dist_matrix, excl_mask], 2)
    chex.assert_equal_shape([dist_matrix, excl_mask])
    chex.assert_rank([r0_grid, b_grid], 1)

    mask = jnp.asarray(excl_mask, dtype=dist_matrix.dtype)

    def for_r0(r0: float) -> Float[Array, "L N_p"]:
        def for_b(b: float) -> Float[Array, "N_p"]:
            contacts = sigmoid_switch(dist_matrix, r0=r0, b=b)
            return jnp.sum(contacts * mask, axis=-1)

        return jax.vmap(for_b)(b_grid)

    return jax.vmap(for_r0)(r0_grid)


def bv_contact_counts(
    coords: Float[Array, "N_atoms 3"],
    amide_N_idx: Int[Array, "N_res"],
    amide_H_idx: Int[Array, "N_res"],
    heavy_idx: Int[Array, "N_heavy"],
    backbone_O_idx: Int[Array, "N_bb_O"],
    excl_mask_c: Float[Array, "N_res N_heavy"] | Array,
    excl_mask_h: Float[Array, "N_res N_bb_O"] | Array,
    x_c: float = 6.5,
    x_h: float = 2.4,
    b: float = 10.0,
) -> tuple[Float[Array, "N_res"], Float[Array, "N_res"]]:
    chex.assert_rank(coords, 2)
    if coords.shape[-1] != 3:
        raise ValueError("coords must have shape (N_atoms, 3)")
    chex.assert_rank([amide_N_idx, amide_H_idx, heavy_idx, backbone_O_idx], 1)
    chex.assert_rank([excl_mask_c, excl_mask_h], 2)

    n_coords = coords[amide_N_idx]
    h_coords = coords[amide_H_idx]
    heavy_coords = coords[heavy_idx]
    o_coords = coords[backbone_O_idx]

    dist_c = dist_matrix_asymmetric(n_coords, heavy_coords)
    dist_h = dist_matrix_asymmetric(h_coords, o_coords)

    mask_c = jnp.asarray(excl_mask_c, dtype=coords.dtype)
    mask_h = jnp.asarray(excl_mask_h, dtype=coords.dtype)
    chex.assert_equal_shape([dist_c, mask_c])
    chex.assert_equal_shape([dist_h, mask_h])

    nc = jnp.sum(sigmoid_switch(dist_c, r0=x_c, b=b) * mask_c, axis=-1)
    nh = jnp.sum(sigmoid_switch(dist_h, r0=x_h, b=b) * mask_h, axis=-1)
    return nc, nh

