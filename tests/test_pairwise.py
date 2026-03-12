import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.pairwise import (
    chunked_dist_apply,
    dist_from_sq_block,
    dist_matrix_asymmetric,
    dist_matrix_block,
)


def test_dist_matrix_asymmetric_known_geometries():
    probe = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    env = jnp.array([[3.0, 4.0, 0.0], [10.0, 0.0, 0.0]], dtype=jnp.float32)
    dist = dist_matrix_asymmetric(probe, env)
    expected = jnp.array([[5.0, 10.0], [4.47213595, 9.0]], dtype=jnp.float32)
    np.testing.assert_allclose(dist, expected, atol=1e-5, rtol=1e-6)


def test_dist_matrix_asymmetric_symmetry_and_self_diagonal():
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    a = jax.random.normal(key_a, (8, 3), dtype=jnp.float32)
    b = jax.random.normal(key_b, (5, 3), dtype=jnp.float32)

    dist_ab = dist_matrix_asymmetric(a, b)
    dist_ba = dist_matrix_asymmetric(b, a)
    np.testing.assert_allclose(dist_ab, dist_ba.T, atol=1e-5, rtol=1e-5)

    dist_self = dist_matrix_asymmetric(a, a)
    np.testing.assert_allclose(jnp.diag(dist_self), 1e-5, atol=1e-6, rtol=1e-6)


def test_dist_matrix_asymmetric_gradient():
    probe = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    env = jnp.array([[3.0, 4.0, 0.0]], dtype=jnp.float32)

    grad = jax.grad(lambda p: jnp.sum(dist_matrix_asymmetric(p, env)))(probe)
    np.testing.assert_allclose(grad[0, 0], -0.6, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(grad[0, 1], -0.8, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(grad[0, 2], 0.0, atol=1e-6, rtol=1e-6)


def test_dist_matrix_block_properties():
    key = jax.random.PRNGKey(42)
    coords = jax.random.normal(key, (5, 3), dtype=jnp.float32)
    dist_sq = dist_matrix_block(coords, coords)

    np.testing.assert_allclose(jnp.diag(dist_sq), 0.0, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(dist_sq, dist_sq.T, atol=1e-7, rtol=1e-7)
    assert jnp.all(dist_sq >= 0.0)


def test_dist_from_sq_block_zero_handling_and_gradient():
    dist_sq = jnp.zeros((3, 3), dtype=jnp.float32)
    dist = dist_from_sq_block(dist_sq)
    np.testing.assert_allclose(dist, jnp.zeros((3, 3), dtype=jnp.float32), atol=1e-7)

    grad = jax.grad(lambda x: jnp.sum(dist_from_sq_block(x)))(dist_sq)
    assert jnp.all(jnp.isfinite(grad))
    np.testing.assert_allclose(grad, 0.0, atol=1e-7, rtol=1e-7)


def test_dist_block_matches_asymmetric():
    key_a, key_b = jax.random.split(jax.random.PRNGKey(7))
    coords_a = jax.random.normal(key_a, (20, 3), dtype=jnp.float32)
    coords_b = jax.random.normal(key_b, (30, 3), dtype=jnp.float32)

    d_asym = dist_matrix_asymmetric(coords_a, coords_b)
    d_block = dist_from_sq_block(dist_matrix_block(coords_a, coords_b))
    np.testing.assert_allclose(d_asym, d_block, atol=1e-4, rtol=1e-4)


def test_chunked_dist_apply_matches_dense():
    key_probe, key_env = jax.random.split(jax.random.PRNGKey(99))
    probe = jax.random.normal(key_probe, (17, 3), dtype=jnp.float32)
    env = jax.random.normal(key_env, (11, 3), dtype=jnp.float32)

    def apply_fn(chunk, env_coords):
        return dist_matrix_asymmetric(chunk, env_coords)

    dense = apply_fn(probe, env)
    chunked = chunked_dist_apply(probe, env, apply_fn=apply_fn, chunk_size=4)

    assert chunked.shape == dense.shape
    np.testing.assert_allclose(chunked, dense, atol=1e-5, rtol=1e-5)
