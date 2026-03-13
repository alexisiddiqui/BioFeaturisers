"""Helpers for single-structure and trajectory forward dispatch."""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def apply_forward(
    forward_fn: Callable[[Float[Array, "n_atoms 3"]], Array],
    coords: Float[Array, "n_atoms 3"] | Float[Array, "n_frames n_atoms 3"],
    weights: Float[Array, "n_frames"] | None = None,
    batch_size: int = 8,
) -> Array:
    """Dispatch a forward function across a single structure or trajectory."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    coords_arr = jnp.asarray(coords)
    if coords_arr.ndim == 2:
        if coords_arr.shape[1] != 3:
            raise ValueError(f"coords must have shape (n_atoms, 3), got {coords_arr.shape}")
        if weights is not None:
            raise ValueError("weights must be None when coords is a single structure")
        return jnp.asarray(forward_fn(coords_arr))

    if coords_arr.ndim != 3 or coords_arr.shape[2] != 3:
        raise ValueError(f"coords must have shape (n_atoms, 3) or (n_frames, n_atoms, 3), got {coords_arr.shape}")

    weights_arr = None
    if weights is not None:
        weights_arr = jnp.asarray(weights, dtype=coords_arr.dtype)
        chex.assert_rank(weights_arr, 1)
        if weights_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError("weights length must match trajectory frame count")

    @jax.checkpoint
    def _per_frame(frame_coords: Float[Array, "n_atoms 3"]) -> Array:
        return jnp.asarray(forward_fn(frame_coords))

    outputs = jax.lax.map(_per_frame, coords_arr, batch_size=batch_size)
    if weights_arr is None:
        return jnp.mean(outputs, axis=0)

    weight_shape = (weights_arr.shape[0],) + (1,) * (outputs.ndim - 1)
    return jnp.sum(outputs * jnp.reshape(weights_arr, weight_shape), axis=0)


def weighted_ensemble(
    forward_fn: Callable[[Float[Array, "n_atoms 3"]], Array],
    coords: Float[Array, "n_frames n_atoms 3"],
    weights: Float[Array, "n_frames"],
    batch_size: int = 8,
) -> Array:
    """Weighted ensemble average wrapper."""
    return apply_forward(forward_fn, coords, weights=weights, batch_size=batch_size)


def uniform_ensemble(
    forward_fn: Callable[[Float[Array, "n_atoms 3"]], Array],
    coords: Float[Array, "n_frames n_atoms 3"],
    batch_size: int = 8,
) -> Array:
    """Uniform ensemble average wrapper."""
    return apply_forward(forward_fn, coords, weights=None, batch_size=batch_size)


__all__ = ["apply_forward", "uniform_ensemble", "weighted_ensemble"]
