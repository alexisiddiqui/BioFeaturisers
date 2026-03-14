"""Public SAXS forward interface wrappers."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from biofeaturisers.config import SAXSConfig

from .features import SAXSFeatures
from .foxs import saxs_forward as _saxs_forward_single
from .foxs import saxs_trajectory as _saxs_forward_trajectory


def forward(
    coords: Float[Array, "n_atoms 3"] | Float[Array, "n_frames n_atoms 3"],
    features: SAXSFeatures,
    config: SAXSConfig | None = None,
    c1: float | None = None,
    c2: float | None = None,
    weights: Float[Array, "n_frames"] | None = None,
) -> Float[Array, "n_q"]:
    """Dispatch SAXS forward computation for single structures or trajectories."""
    cfg = config or SAXSConfig()
    c1_value = cfg.c1 if c1 is None else c1
    c2_value = cfg.c2 if c2 is None else c2

    coords_arr = jnp.asarray(coords, dtype=jnp.float32)
    if coords_arr.ndim == 2:
        return _saxs_forward_single(
            coords=coords_arr,
            features=features,
            c1=float(c1_value),
            c2=float(c2_value),
            chunk_size=int(cfg.chunk_size),
        )
    if coords_arr.ndim == 3:
        return _saxs_forward_trajectory(
            trajectory=coords_arr,
            features=features,
            c1=float(c1_value),
            c2=float(c2_value),
            weights=weights,
            batch_size=int(cfg.batch_size),
            chunk_size=int(cfg.chunk_size),
        )
    raise ValueError(
        f"coords must have shape (n_atoms, 3) or (n_frames, n_atoms, 3), got {tuple(coords_arr.shape)}"
    )


__all__ = ["forward"]

