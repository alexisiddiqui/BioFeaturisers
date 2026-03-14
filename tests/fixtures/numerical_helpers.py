"""Reusable numerical helpers for regression-style tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.safe_math import safe_sinc


def make_linear_coords(n_atoms: int, spacing: float = 1.5) -> jax.Array:
    """Create deterministic collinear coordinates along x-axis."""
    if n_atoms <= 0:
        raise ValueError("n_atoms must be > 0")
    x = jnp.arange(n_atoms, dtype=jnp.float32) * jnp.float32(spacing)
    return jnp.stack((x, jnp.zeros_like(x), jnp.zeros_like(x)), axis=-1)


def dense_debye_reference(
    coords: jax.Array,
    ff: jax.Array,
    q_values: jax.Array,
) -> jax.Array:
    """Naive ``O(N^2)`` Debye reference used for testing chunked kernels."""
    coords_arr = jnp.asarray(coords, dtype=jnp.float32)
    ff_arr = jnp.asarray(ff, dtype=jnp.float32)
    q_arr = jnp.asarray(q_values, dtype=jnp.float32)
    diff = coords_arr[:, None, :] - coords_arr[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    qr = q_arr[None, None, :] * dist[:, :, None]
    return jnp.sum(ff_arr[:, None, :] * ff_arr[None, :, :] * safe_sinc(qr), axis=(0, 1))


def assert_directional_gradient_close(
    loss_fn,
    x: jax.Array,
    *,
    eps: float = 1e-4,
    rtol: float = 1e-2,
    atol: float = 1e-5,
    n_dirs: int = 5,
    seed: int = 0,
) -> None:
    """Compare autodiff directional derivatives with central finite differences."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    grad = jax.grad(loss_fn)(x_arr)
    rng = np.random.default_rng(seed)

    for _ in range(n_dirs):
        direction = rng.normal(size=x_arr.shape).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        direction = direction / norm
        direction_arr = jnp.asarray(direction, dtype=jnp.float32)

        fd = (
            loss_fn(x_arr + eps * direction_arr) - loss_fn(x_arr - eps * direction_arr)
        ) / (2.0 * eps)
        ad = jnp.sum(grad * direction_arr)
        np.testing.assert_allclose(float(fd), float(ad), rtol=rtol, atol=atol)

