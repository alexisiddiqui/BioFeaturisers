"""Shared pairwise distance kernels."""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .safe_math import safe_sqrt, safe_sqrt_sym


def _validate_xyz(coords: Array, name: str) -> Array:
    coords_arr = jnp.asarray(coords)
    chex.assert_rank(coords_arr, 2)
    if coords_arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {coords_arr.shape}")
    return coords_arr


def dist_matrix_asymmetric(
    probe_coords: Float[Array, "n_probe 3"],
    env_coords: Float[Array, "n_env 3"],
    eps: float = 1e-10,
) -> Float[Array, "n_probe n_env"]:
    """Compute ``||probe_i - env_j||`` using the matmul identity."""
    probe = _validate_xyz(probe_coords, "probe_coords")
    env = _validate_xyz(env_coords, "env_coords")

    probe_sq = jnp.sum(probe * probe, axis=1, keepdims=True)
    env_sq = jnp.sum(env * env, axis=1, keepdims=True)
    cross = probe @ env.T
    dist_sq = jnp.maximum(0.0, probe_sq - 2.0 * cross + env_sq.T)
    return safe_sqrt(dist_sq, eps=eps)


def dist_matrix_block(
    coords_i: Float[Array, "n_i 3"],
    coords_j: Float[Array, "n_j 3"],
) -> Float[Array, "n_i n_j"]:
    """Compute squared distances with explicit broadcasting."""
    coords_i_arr = _validate_xyz(coords_i, "coords_i")
    coords_j_arr = _validate_xyz(coords_j, "coords_j")
    diff = coords_i_arr[:, None, :] - coords_j_arr[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def dist_from_sq_block(dist_sq: Float[Array, "n_i n_j"]) -> Float[Array, "n_i n_j"]:
    """Convert squared distances to distances with zero-safe gradients."""
    return safe_sqrt_sym(jnp.asarray(dist_sq))


def chunked_dist_apply(
    probe_coords: Float[Array, "n_probe 3"],
    env_coords: Float[Array, "n_env 3"],
    apply_fn: Callable[[Array, Array], Array],
    chunk_size: int = 256,
) -> Array:
    """Apply a pairwise function over probe chunks to bound peak memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    probe = _validate_xyz(probe_coords, "probe_coords")
    env = _validate_xyz(env_coords, "env_coords")

    n_probe = probe.shape[0]
    pad = (-n_probe) % chunk_size
    probe_padded = jnp.pad(probe, ((0, pad), (0, 0)))
    n_chunks = probe_padded.shape[0] // chunk_size
    chunks = probe_padded.reshape((n_chunks, chunk_size, 3))

    @jax.checkpoint
    def _scan_body(_: None, chunk: Array) -> tuple[None, Array]:
        return None, jnp.asarray(apply_fn(chunk, env))

    _, chunk_outputs = jax.lax.scan(_scan_body, None, chunks)
    flat_outputs = jnp.reshape(chunk_outputs, (-1, *chunk_outputs.shape[2:]))
    return flat_outputs[:n_probe]


__all__ = [
    "chunked_dist_apply",
    "dist_from_sq_block",
    "dist_matrix_asymmetric",
    "dist_matrix_block",
]
