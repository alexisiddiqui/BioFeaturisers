"""Hydration parameter fitting for FoXS SAXS profiles."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from biofeaturisers.config import SAXSConfig

from .foxs import saxs_combine


def scaled_reduced_chi2(
    i_calc: Float[Array, "n_q"],
    i_exp: Float[Array, "n_q"],
    sigma: Float[Array, "n_q"],
) -> Float[Array, ""]:
    """Compute reduced ``chi^2`` with analytic intensity scaling."""
    i_calc_arr = jnp.asarray(i_calc, dtype=jnp.float32)
    i_exp_arr = jnp.asarray(i_exp, dtype=jnp.float32)
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)
    if i_calc_arr.ndim != 1 or i_exp_arr.ndim != 1 or sigma_arr.ndim != 1:
        raise ValueError("i_calc, i_exp, and sigma must be rank-1 arrays")
    if i_calc_arr.shape != i_exp_arr.shape or i_calc_arr.shape != sigma_arr.shape:
        raise ValueError("i_calc, i_exp, and sigma must have matching shapes")
    if bool(jnp.any(sigma_arr <= 0.0)):
        raise ValueError("sigma entries must be > 0")

    inv_var = 1.0 / (sigma_arr**2)
    numerator = jnp.sum(i_calc_arr * i_exp_arr * inv_var)
    denominator = jnp.sum((i_calc_arr**2) * inv_var)
    scale = numerator / jnp.maximum(denominator, jnp.asarray(1e-12, dtype=i_calc_arr.dtype))

    residuals = (scale * i_calc_arr - i_exp_arr) / sigma_arr
    dof = jnp.maximum(i_exp_arr.shape[0] - 1, 1)
    return jnp.sum(residuals * residuals) / jnp.asarray(dof, dtype=i_calc_arr.dtype)


def fit_c1_c2(
    partials: Float[Array, "6 n_q"],
    i_exp: Float[Array, "n_q"],
    sigma: Float[Array, "n_q"],
    config: SAXSConfig,
) -> tuple[float, float, float]:
    """Grid-search ``c1``/``c2`` over cached partials."""
    partials_arr = jnp.asarray(partials, dtype=jnp.float32)
    i_exp_arr = jnp.asarray(i_exp, dtype=jnp.float32)
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)
    if partials_arr.ndim != 2 or int(partials_arr.shape[0]) != 6:
        raise ValueError(f"partials must have shape (6, n_q), got {tuple(partials_arr.shape)}")
    if i_exp_arr.shape[0] != partials_arr.shape[1]:
        raise ValueError("i_exp length must match partial q-grid length")
    if sigma_arr.shape[0] != partials_arr.shape[1]:
        raise ValueError("sigma length must match partial q-grid length")

    c1_grid = jnp.linspace(config.c1_range[0], config.c1_range[1], config.c1_steps, dtype=jnp.float32)
    c2_grid = jnp.linspace(config.c2_range[0], config.c2_range[1], config.c2_steps, dtype=jnp.float32)

    def _chi2_for(c1: Float[Array, ""], c2: Float[Array, ""]) -> Float[Array, ""]:
        i_calc = saxs_combine(partials_arr, c1=c1, c2=c2)
        return scaled_reduced_chi2(i_calc=i_calc, i_exp=i_exp_arr, sigma=sigma_arr)

    chi2_grid = jax.vmap(jax.vmap(_chi2_for, in_axes=(None, 0)), in_axes=(0, None))(c1_grid, c2_grid)
    best_flat = jnp.argmin(chi2_grid)
    i_idx, j_idx = jnp.unravel_index(best_flat, chi2_grid.shape)
    return (
        float(c1_grid[i_idx]),
        float(c2_grid[j_idx]),
        float(chi2_grid[i_idx, j_idx]),
    )


def fit_c1_c2_analytic(
    partials: Float[Array, "6 n_q"],
    i_exp: Float[Array, "n_q"],
    sigma: Float[Array, "n_q"],
    config: SAXSConfig,
) -> tuple[float, float, float]:
    """Alias for explicit naming in hydration fitting APIs."""
    return fit_c1_c2(partials=partials, i_exp=i_exp, sigma=sigma, config=config)


__all__ = ["fit_c1_c2", "fit_c1_c2_analytic", "scaled_reduced_chi2"]

