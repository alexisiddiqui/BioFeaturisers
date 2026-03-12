import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.ensemble import apply_forward


def _vector_forward(coords):
    return jnp.sum(coords, axis=0)


def _scalar_forward(coords):
    return jnp.sum(coords)


def test_apply_forward_single_structure_passthrough():
    coords = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
    result = apply_forward(_vector_forward, coords, weights=None)
    expected = jnp.sum(coords, axis=0)
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_apply_forward_uniform_trajectory_mean():
    key = jax.random.PRNGKey(0)
    traj = jax.random.normal(key, (5, 10, 3), dtype=jnp.float32)

    result = apply_forward(_vector_forward, traj, weights=None, batch_size=2)
    expected = jnp.mean(jax.vmap(_vector_forward)(traj), axis=0)
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_apply_forward_weighted_trajectory_mean():
    key = jax.random.PRNGKey(1)
    traj = jax.random.normal(key, (5, 10, 3), dtype=jnp.float32)
    weights = jnp.array([0.5, 0.3, 0.1, 0.05, 0.05], dtype=jnp.float32)

    result = apply_forward(_vector_forward, traj, weights=weights, batch_size=2)
    expected = jnp.sum(weights[:, None] * jax.vmap(_vector_forward)(traj), axis=0)
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_apply_forward_weighted_scalar_output():
    key = jax.random.PRNGKey(2)
    traj = jax.random.normal(key, (4, 6, 3), dtype=jnp.float32)
    weights = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)

    result = apply_forward(_scalar_forward, traj, weights=weights, batch_size=2)
    expected = jnp.sum(weights * jax.vmap(_scalar_forward)(traj))
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_apply_forward_gradient_through_trajectory():
    key = jax.random.PRNGKey(3)
    traj = jax.random.normal(key, (5, 8, 3), dtype=jnp.float32)
    loss = lambda x: jnp.sum(apply_forward(_vector_forward, x, weights=None, batch_size=2))

    grad = jax.grad(loss)(traj)
    assert grad.shape == traj.shape
    assert jnp.all(jnp.isfinite(grad))

