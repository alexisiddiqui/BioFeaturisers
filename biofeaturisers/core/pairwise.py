from __future__ import annotations

from collections.abc import Callable
from functools import partial

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .safe_math import safe_sqrt


def _assert_xyz(coords: Float[Array, "N 3"], name: str) -> None:
    chex.assert_rank(coords, 2)
    if coords.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")


def dist_matrix_asymmetric(
    probe_coords: Float[Array, "N_p 3"],
    env_coords: Float[Array, "N_e 3"],
    eps: float = 1e-10,
) -> Float[Array, "N_p N_e"]:
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive")
    _assert_xyz(probe_coords, "probe_coords")
    _assert_xyz(env_coords, "env_coords")

    p_sq = jnp.sum(probe_coords**2, axis=1, keepdims=True)
    e_sq = jnp.sum(env_coords**2, axis=1, keepdims=True)
    cross = probe_coords @ env_coords.T
    dist_sq = jnp.maximum(0.0, p_sq - 2.0 * cross + e_sq.T)

    # Reduce cancellation error on the i==j diagonal when the two sets share length.
    if probe_coords.shape[0] == env_coords.shape[0]:
        diag_sq = jnp.sum((probe_coords - env_coords) ** 2, axis=-1)
        diag_idx = jnp.arange(probe_coords.shape[0])
        dist_sq = dist_sq.at[diag_idx, diag_idx].set(jnp.maximum(0.0, diag_sq))

    return safe_sqrt(dist_sq, eps=eps)


def dist_matrix_block(
    coords_i: Float[Array, "B_i 3"],
    coords_j: Float[Array, "B_j 3"],
) -> Float[Array, "B_i B_j"]:
    _assert_xyz(coords_i, "coords_i")
    _assert_xyz(coords_j, "coords_j")

    diff = coords_i[:, None, :] - coords_j[None, :, :]
    return jnp.sum(diff**2, axis=-1)


def dist_from_sq_block(dist_sq: Float[Array, "B_i B_j"]) -> Float[Array, "B_i B_j"]:
    chex.assert_rank(dist_sq, 2)
    is_zero = dist_sq <= 0.0
    safe_sq = jnp.where(is_zero, 1.0, dist_sq)
    return jnp.where(is_zero, 0.0, jnp.sqrt(safe_sq))


@partial(jax.jit, static_argnames=("apply_fn", "chunk_size"))
def chunked_dist_apply(
    probe_coords: Float[Array, "N_p 3"],
    env_coords: Float[Array, "N_e 3"],
    apply_fn: Callable[[Float[Array, "C 3"], Float[Array, "N_e 3"]], Array],
    chunk_size: int = 256,
) -> Array:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    _assert_xyz(probe_coords, "probe_coords")
    _assert_xyz(env_coords, "env_coords")

    n_probe = probe_coords.shape[0]
    pad = (-n_probe) % chunk_size
    probe_padded = jnp.pad(probe_coords, ((0, pad), (0, 0)))

    n_chunks = probe_padded.shape[0] // chunk_size
    chunks = probe_padded.reshape(n_chunks, chunk_size, 3)

    @jax.checkpoint
    def body(_: None, chunk: Float[Array, "C 3"]) -> tuple[None, Array]:
        return None, apply_fn(chunk, env_coords)

    _, results = jax.lax.scan(body, None, chunks)
    flat = results.reshape((results.shape[0] * results.shape[1],) + results.shape[2:])
    return flat[:n_probe]
