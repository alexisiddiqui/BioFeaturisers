"""Tests for trajectory forward dispatch."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from biofeaturisers.core.ensemble import apply_forward


def _vector_forward(coords: jax.Array) -> jax.Array:
    return jnp.sum(coords, axis=0)


def _scalar_forward(coords: jax.Array) -> jax.Array:
    return jnp.sum(coords)


def test_apply_forward_single_structure_passthrough() -> None:
    coords = jax.random.normal(jax.random.PRNGKey(0), (10, 3), dtype=jnp.float32)
    result = apply_forward(_vector_forward, coords, weights=None)
    expected = jnp.sum(coords, axis=0)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_apply_forward_uniform_trajectory_mean() -> None:
    traj = jax.random.normal(jax.random.PRNGKey(1), (5, 10, 3), dtype=jnp.float32)
    result = apply_forward(_vector_forward, traj, weights=None, batch_size=2)
    expected = jnp.mean(jax.vmap(_vector_forward)(traj), axis=0)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_apply_forward_weighted_trajectory_mean() -> None:
    traj = jax.random.normal(jax.random.PRNGKey(2), (5, 10, 3), dtype=jnp.float32)
    weights = jnp.asarray([0.5, 0.3, 0.1, 0.05, 0.05], dtype=jnp.float32)
    result = apply_forward(_vector_forward, traj, weights=weights, batch_size=2)
    expected = jnp.sum(weights[:, None] * jax.vmap(_vector_forward)(traj), axis=0)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_apply_forward_scalar_output_weighted() -> None:
    traj = jax.random.normal(jax.random.PRNGKey(3), (4, 6, 3), dtype=jnp.float32)
    weights = jnp.asarray([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
    result = apply_forward(_scalar_forward, traj, weights=weights, batch_size=2)
    expected = jnp.sum(weights * jax.vmap(_scalar_forward)(traj))
    assert np.isclose(float(result), float(expected), atol=1e-6)


def test_apply_forward_gradient_through_trajectory() -> None:
    traj = jax.random.normal(jax.random.PRNGKey(4), (6, 8, 3), dtype=jnp.float32)
    loss = lambda c: jnp.sum(apply_forward(_vector_forward, c, weights=None, batch_size=2))
    grad = jax.grad(loss)(traj)
    assert grad.shape == traj.shape
    assert np.isfinite(np.asarray(grad)).all()


def test_apply_forward_validates_inputs() -> None:
    coords = jnp.zeros((10, 3), dtype=jnp.float32)
    with pytest.raises(ValueError):
        apply_forward(_vector_forward, coords, weights=jnp.ones((2,), dtype=jnp.float32))
    with pytest.raises(ValueError):
        apply_forward(_vector_forward, coords, weights=None, batch_size=0)
