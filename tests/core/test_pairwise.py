"""Tests for pairwise distance kernels."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from biofeaturisers.core.pairwise import (
    chunked_dist_apply,
    dist_from_sq_block,
    dist_matrix_asymmetric,
    dist_matrix_block,
)


def test_dist_matrix_asymmetric_known_geometry() -> None:
    probe = jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    env = jnp.asarray([[3.0, 4.0, 0.0]], dtype=jnp.float32)
    dist = dist_matrix_asymmetric(probe, env)
    assert isinstance(dist, jax.Array)
    chex.assert_shape(dist, (1, 1))
    assert np.isclose(float(dist[0, 0]), 5.0, atol=1e-6)


def test_dist_matrix_asymmetric_symmetry_and_self_distance_floor() -> None:
    a = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=jnp.float32)
    b = jnp.asarray([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]], dtype=jnp.float32)
    d_ab = dist_matrix_asymmetric(a, b)
    d_ba = dist_matrix_asymmetric(b, a)
    np.testing.assert_allclose(np.asarray(d_ab), np.asarray(d_ba).T, atol=1e-6)

    d_self = np.asarray(dist_matrix_asymmetric(a, a))
    np.testing.assert_allclose(np.diag(d_self), np.full((2,), 1e-5, dtype=np.float32), atol=1e-7)


def test_dist_matrix_asymmetric_gradient_matches_expected_direction() -> None:
    probe = jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    env = jnp.asarray([[3.0, 4.0, 0.0]], dtype=jnp.float32)
    grad = jax.grad(lambda p: jnp.sum(dist_matrix_asymmetric(p, env)))(probe)
    np.testing.assert_allclose(np.asarray(grad[0]), np.asarray([-0.6, -0.8, 0.0]), atol=1e-6)


def test_dist_matrix_block_and_dist_from_sq_block_contract() -> None:
    coords = jax.random.normal(jax.random.PRNGKey(0), (5, 3), dtype=jnp.float32)
    dsq = dist_matrix_block(coords, coords)
    dsq_np = np.asarray(dsq)
    np.testing.assert_allclose(np.diag(dsq_np), np.zeros((5,), dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(dsq_np, dsq_np.T, atol=1e-6)
    assert (dsq_np >= -1e-7).all()

    d = dist_from_sq_block(jnp.zeros((3, 3), dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(d), np.zeros((3, 3), dtype=np.float32))
    g = jax.grad(lambda x: jnp.sum(dist_from_sq_block(x)))(jnp.zeros((3, 3), dtype=jnp.float32))
    assert np.isfinite(np.asarray(g)).all()


def test_block_and_asymmetric_distance_agree() -> None:
    coords_a = jax.random.normal(jax.random.PRNGKey(1), (20, 3), dtype=jnp.float32)
    coords_b = jax.random.normal(jax.random.PRNGKey(2), (30, 3), dtype=jnp.float32)

    d_asym = dist_matrix_asymmetric(coords_a, coords_b)
    d_block = dist_from_sq_block(dist_matrix_block(coords_a, coords_b))
    np.testing.assert_allclose(np.asarray(d_asym), np.asarray(d_block), atol=1e-4, rtol=1e-5)


def test_chunked_dist_apply_matches_dense_result() -> None:
    probe = jax.random.normal(jax.random.PRNGKey(3), (11, 3), dtype=jnp.float32)
    env = jax.random.normal(jax.random.PRNGKey(4), (7, 3), dtype=jnp.float32)

    dense = dist_matrix_asymmetric(probe, env)
    chunked = chunked_dist_apply(probe, env, dist_matrix_asymmetric, chunk_size=4)
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(dense), atol=1e-6, rtol=1e-6)


def test_chunked_dist_apply_validates_chunk_size() -> None:
    probe = jnp.zeros((2, 3), dtype=jnp.float32)
    env = jnp.zeros((3, 3), dtype=jnp.float32)
    with pytest.raises(ValueError):
        chunked_dist_apply(probe, env, dist_matrix_asymmetric, chunk_size=0)
