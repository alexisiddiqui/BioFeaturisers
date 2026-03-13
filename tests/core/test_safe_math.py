"""Tests for numerically safe math primitives."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.safe_math import (
    diagonal_self_pairs,
    safe_mask,
    safe_sinc,
    safe_sqrt,
    safe_sqrt_sym,
)


def _safe_sinc_no_custom_vjp(qr: jax.Array) -> jax.Array:
    large = jnp.abs(qr) > 1e-8
    safe_qr = jnp.where(large, qr, jnp.ones_like(qr))
    normal = jnp.sin(safe_qr) / safe_qr
    taylor = 1.0 - qr**2 / 6.0 + qr**4 / 120.0
    return jnp.where(large, normal, taylor)


def test_safe_sinc_forward_values() -> None:
    values = jnp.asarray([0.0, 1e-10, 1e-8, 0.1, jnp.pi, 2.0 * jnp.pi, 100.0], dtype=jnp.float32)
    out = np.asarray(safe_sinc(values))

    expected = np.asarray(
        [
            1.0,
            1.0,
            float(np.sin(1e-8) / 1e-8),
            float(np.sin(0.1) / 0.1),
            float(np.sin(np.pi) / np.pi),
            float(np.sin(2.0 * np.pi) / (2.0 * np.pi)),
            float(np.sin(100.0) / 100.0),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_safe_sinc_gradients_origin_and_finite() -> None:
    grad_scalar = jax.grad(lambda x: safe_sinc(x))
    g0 = float(grad_scalar(jnp.float32(0.0)))
    assert np.isfinite(g0)
    assert abs(g0) < 1e-6

    qr = jnp.asarray([0.0, 1e-10, 0.01, 0.5, 1.0, jnp.pi, 50.0], dtype=jnp.float32)
    g_batch = jax.grad(lambda x: jnp.sum(safe_sinc(x)))(qr)
    assert np.isfinite(np.asarray(g_batch)).all()


def test_safe_sinc_custom_vjp_matches_naive_autodiff() -> None:
    qr = jnp.asarray([0.0, 1e-9, 0.01, 0.5, 1.0, jnp.pi, 5.0, 50.0], dtype=jnp.float32)
    g_custom = jax.grad(lambda x: jnp.sum(safe_sinc(x)))(qr)
    g_naive = jax.grad(lambda x: jnp.sum(_safe_sinc_no_custom_vjp(x)))(qr)
    np.testing.assert_allclose(np.asarray(g_custom), np.asarray(g_naive), atol=1e-5, rtol=1e-5)


def test_safe_sqrt_and_safe_sqrt_sym_values() -> None:
    assert np.isclose(float(safe_sqrt(jnp.float32(0.0))), 1e-5, atol=1e-8)
    assert np.isclose(float(safe_sqrt(jnp.float32(4.0))), 2.0, atol=1e-6)

    z = jnp.zeros((4, 4), dtype=jnp.float32)
    eye = jnp.eye(4, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(safe_sqrt_sym(z)), np.zeros((4, 4), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(safe_sqrt_sym(eye)), np.eye(4, dtype=np.float32))


def test_safe_sqrt_sym_gradients_are_finite_and_correct() -> None:
    z = jnp.zeros((4, 4), dtype=jnp.float32)
    g_zero = jax.grad(lambda x: jnp.sum(safe_sqrt_sym(x)))(z)
    assert np.isfinite(np.asarray(g_zero)).all()

    d = jnp.eye(4, dtype=jnp.float32) * 4.0
    g = np.asarray(jax.grad(lambda x: jnp.sum(safe_sqrt_sym(x)))(d))
    np.testing.assert_allclose(np.diag(g), np.full((4,), 0.25, dtype=np.float32), atol=1e-6)
    off_diag = g.copy()
    np.fill_diagonal(off_diag, 0.0)
    np.testing.assert_allclose(off_diag, np.zeros((4, 4), dtype=np.float32), atol=1e-6)


def test_safe_mask_masks_values_and_keeps_gradients_finite() -> None:
    mask = jnp.asarray([True, False, True, False], dtype=jnp.bool_)
    x = jnp.asarray([4.0, 0.0, 9.0, 0.0], dtype=jnp.float32)
    result = safe_mask(mask, jnp.sqrt, x, placeholder=0.0)
    np.testing.assert_allclose(np.asarray(result), np.asarray([2.0, 0.0, 3.0, 0.0], dtype=np.float32))

    grad = jax.grad(lambda v: jnp.sum(safe_mask(mask, jnp.sqrt, v)))(x)
    assert np.isfinite(np.asarray(grad)).all()
    np.testing.assert_allclose(np.asarray(grad)[~np.asarray(mask)], np.asarray([0.0, 0.0]), atol=1e-7)


def test_diagonal_self_pairs_contract() -> None:
    ff = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    result = diagonal_self_pairs(ff)
    assert isinstance(result, jax.Array)
    chex.assert_shape(result, (2,))
    np.testing.assert_allclose(np.asarray(result), np.asarray([35.0, 56.0], dtype=np.float32))
