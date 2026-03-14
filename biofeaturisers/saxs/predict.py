"""End-to-end SAXS prediction wrapper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from biotite.structure import AtomArray
from jaxtyping import Array, Float

from biofeaturisers.config import SAXSConfig
from biofeaturisers.core.output_index import OutputIndex

from .debye import saxs_six_partials
from .featurise import featurise
from .foxs import saxs_combine
from .hydration import fit_c1_c2, scaled_reduced_chi2


def _mean_partials(
    coords: Float[Array, "n_atoms 3"] | Float[Array, "n_frames n_atoms 3"],
    features,
    config: SAXSConfig,
    weights: Float[Array, "n_frames"] | None,
) -> jax.Array:
    coords_arr = jnp.asarray(coords, dtype=jnp.float32)
    if coords_arr.ndim == 2:
        return saxs_six_partials(coords_arr, features, chunk_size=int(config.chunk_size))
    if coords_arr.ndim != 3:
        raise ValueError(
            f"coords must have shape (n_atoms, 3) or (n_frames, n_atoms, 3), got {tuple(coords_arr.shape)}"
        )

    @jax.checkpoint
    def _per_frame(frame: Float[Array, "n_atoms 3"]) -> jax.Array:
        return saxs_six_partials(frame, features, chunk_size=int(config.chunk_size))

    frame_partials = jax.lax.map(_per_frame, coords_arr, batch_size=int(config.batch_size))
    if weights is None:
        return jnp.mean(frame_partials, axis=0)

    w = jnp.asarray(weights, dtype=frame_partials.dtype)
    if w.ndim != 1 or int(w.shape[0]) != int(frame_partials.shape[0]):
        raise ValueError("weights must be rank-1 and match trajectory frame count")
    return jnp.sum(w[:, None, None] * frame_partials, axis=0)


def predict(
    atom_array: AtomArray,
    config: SAXSConfig | None = None,
    output_index: OutputIndex | None = None,
    coords: Float[Array, "n_atoms 3"] | Float[Array, "n_frames n_atoms 3"] | None = None,
    weights: Float[Array, "n_frames"] | None = None,
    i_exp: Float[Array, "n_q"] | None = None,
    sigma: Float[Array, "n_q"] | None = None,
) -> jax.Array | tuple[jax.Array, float, float, float]:
    """Run ``featurise -> forward`` and optionally fit hydration parameters."""
    cfg = config or SAXSConfig()
    features = featurise(atom_array=atom_array, config=cfg, output_index=output_index)

    if coords is None:
        coords_arr = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))
    else:
        coords_arr = jnp.asarray(coords, dtype=jnp.float32)

    partials = _mean_partials(coords=coords_arr, features=features, config=cfg, weights=weights)

    if i_exp is None:
        return saxs_combine(partials=partials, c1=cfg.c1, c2=cfg.c2)

    i_exp_arr = jnp.asarray(i_exp, dtype=jnp.float32)
    if i_exp_arr.ndim != 1 or int(i_exp_arr.shape[0]) != int(features.q_values.shape[0]):
        raise ValueError("i_exp must be rank-1 and match SAXSFeatures q-grid length")

    sigma_arr = (
        jnp.ones_like(i_exp_arr, dtype=jnp.float32)
        if sigma is None
        else jnp.asarray(sigma, dtype=jnp.float32)
    )
    if sigma_arr.shape != i_exp_arr.shape:
        raise ValueError("sigma must match i_exp shape")

    if cfg.fit_c1_c2:
        c1_opt, c2_opt, chi2 = fit_c1_c2(partials=partials, i_exp=i_exp_arr, sigma=sigma_arr, config=cfg)
        i_q = saxs_combine(partials=partials, c1=c1_opt, c2=c2_opt)
        return i_q, chi2, c1_opt, c2_opt

    i_q = saxs_combine(partials=partials, c1=cfg.c1, c2=cfg.c2)
    chi2 = float(scaled_reduced_chi2(i_calc=i_q, i_exp=i_exp_arr, sigma=sigma_arr))
    return i_q, chi2, float(cfg.c1), float(cfg.c2)


__all__ = ["predict"]

