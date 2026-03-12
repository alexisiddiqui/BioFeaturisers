from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def apply_forward(
    forward_fn: Callable[[Float[Array, "N 3"]], Array],
    coords: Float[Array, "N 3"] | Float[Array, "T N 3"],
    weights: Float[Array, "T"] | None = None,
    batch_size: int = 8,
) -> Array:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if coords.ndim == 2:
        chex.assert_rank(coords, 2)
        if weights is not None:
            raise ValueError("weights must be None for single-structure input")
        return forward_fn(coords)
    if coords.ndim != 3:
        raise ValueError("coords must have shape (N, 3) or (T, N, 3)")

    chex.assert_rank(coords, 3)

    @jax.checkpoint
    def per_frame(coords_t: Float[Array, "N 3"]) -> Array:
        return forward_fn(coords_t)

    all_outputs = jax.lax.map(per_frame, coords, batch_size=batch_size)
    if weights is None:
        return jnp.mean(all_outputs, axis=0)

    weights_array = jnp.asarray(weights, dtype=all_outputs.dtype)
    chex.assert_rank(weights_array, 1)
    if weights_array.shape[0] != coords.shape[0]:
        raise ValueError("weights length must match number of trajectory frames")

    broadcast_shape = (weights_array.shape[0],) + (1,) * (all_outputs.ndim - 1)
    weighted = weights_array.reshape(broadcast_shape) * all_outputs
    return jnp.sum(weighted, axis=0)
