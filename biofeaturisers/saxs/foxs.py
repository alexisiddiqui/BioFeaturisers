"""FoXS recombination and forward helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .debye import saxs_six_partials
from .features import SAXSFeatures


def saxs_combine(
    partials: Float[Array, "6 n_q"],
    c1: float,
    c2: float,
) -> Float[Array, "n_q"]:
    """Recombine six FoXS partials into ``I(q)``."""
    partials_arr = jnp.asarray(partials, dtype=jnp.float32)
    if partials_arr.ndim != 2 or int(partials_arr.shape[0]) != 6:
        raise ValueError(f"partials must have shape (6, n_q), got {tuple(partials_arr.shape)}")
    i_aa, i_cc, i_ss, i_ac, i_as, i_cs = partials_arr
    c1_arr = jnp.asarray(c1, dtype=partials_arr.dtype)
    c2_arr = jnp.asarray(c2, dtype=partials_arr.dtype)
    return i_aa - c1_arr * i_ac + (c1_arr**2) * i_cc + c2_arr * i_as - c1_arr * c2_arr * i_cs + (c2_arr**2) * i_ss


def saxs_forward(
    coords: Float[Array, "n_atoms 3"],
    features: SAXSFeatures,
    c1: float = 1.0,
    c2: float = 0.0,
    chunk_size: int = 512,
) -> Float[Array, "n_q"]:
    """Compute a single-structure SAXS profile."""
    partials = saxs_six_partials(coords=coords, features=features, chunk_size=chunk_size)
    return saxs_combine(partials, c1=c1, c2=c2)


def saxs_trajectory(
    trajectory: Float[Array, "n_frames n_atoms 3"],
    features: SAXSFeatures,
    c1: float = 1.0,
    c2: float = 0.0,
    weights: Float[Array, "n_frames"] | None = None,
    batch_size: int = 4,
    chunk_size: int = 512,
) -> Float[Array, "n_q"]:
    """Compute ensemble-averaged SAXS profile from a trajectory."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    traj = jnp.asarray(trajectory, dtype=jnp.float32)
    if traj.ndim != 3 or int(traj.shape[2]) != 3:
        raise ValueError(
            f"trajectory must have shape (n_frames, n_atoms, 3), got {tuple(traj.shape)}"
        )

    @jax.checkpoint
    def _per_frame(frame_coords: Float[Array, "n_atoms 3"]) -> Float[Array, "6 n_q"]:
        return saxs_six_partials(coords=frame_coords, features=features, chunk_size=chunk_size)

    all_partials = jax.lax.map(_per_frame, traj, batch_size=batch_size)
    if weights is None:
        mean_partials = jnp.mean(all_partials, axis=0)
    else:
        w = jnp.asarray(weights, dtype=all_partials.dtype)
        if w.ndim != 1 or int(w.shape[0]) != int(all_partials.shape[0]):
            raise ValueError("weights must be rank-1 and match trajectory frame count")
        mean_partials = jnp.sum(w[:, None, None] * all_partials, axis=0)
    return saxs_combine(mean_partials, c1=c1, c2=c2)


__all__ = ["saxs_combine", "saxs_forward", "saxs_trajectory"]

