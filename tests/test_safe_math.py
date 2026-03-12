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


def _fd_grad(fn, x: float, eps: float = 1e-4) -> float:
    return float((fn(x + eps) - fn(x - eps)) / (2.0 * eps))


def test_safe_sinc_forward_values():
    qr = jnp.array([0.0, 1e-10, 1e-8, 0.1, jnp.pi, 2.0 * jnp.pi, 100.0], dtype=jnp.float32)
    result = safe_sinc(qr)
    expected = jnp.where(
        jnp.abs(qr) > 1e-8,
        jnp.sin(qr) / qr,
        1.0 - qr**2 / 6.0 + qr**4 / 120.0,
    )
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_safe_sinc_gradients_are_finite():
    g0 = jax.grad(lambda x: safe_sinc(x))(jnp.float32(0.0))
    assert abs(float(g0)) < 1e-7

    qr = jnp.array([0.0, 1e-9, 0.01, 1.0, jnp.pi, 10.0, 100.0], dtype=jnp.float32)
    g = jax.grad(lambda x: jnp.sum(safe_sinc(x)))(qr)
    assert jnp.all(jnp.isfinite(g))


def test_safe_sinc_grad_matches_analytic_and_fd():
    x = jnp.float32(0.1)
    grad_auto = jax.grad(lambda t: safe_sinc(t))(x)
    grad_expected = (jnp.cos(x) - jnp.sin(x) / x) / x
    np.testing.assert_allclose(grad_auto, grad_expected, atol=1e-5, rtol=1e-5)

    fn = lambda t: safe_sinc(jnp.float32(t))
    fd = _fd_grad(fn, float(jnp.pi))
    grad_pi = float(jax.grad(lambda t: safe_sinc(t))(jnp.float32(jnp.pi)))
    np.testing.assert_allclose(grad_pi, fd, atol=1e-4, rtol=1e-3)


def test_safe_sqrt_and_safe_sqrt_sym_behaviour():
    np.testing.assert_allclose(safe_sqrt(jnp.float32(0.0)), 1e-5, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(safe_sqrt(jnp.float32(4.0)), 2.0, atol=1e-7, rtol=1e-6)

    zeros = jnp.zeros((4, 4), dtype=jnp.float32)
    np.testing.assert_allclose(safe_sqrt_sym(zeros), zeros, atol=1e-7)

    eye = jnp.eye(4, dtype=jnp.float32) * 4.0
    grad_eye = jax.grad(lambda m: jnp.sum(safe_sqrt_sym(m)))(eye)
    np.testing.assert_allclose(np.diag(grad_eye), 0.25, atol=1e-6, rtol=1e-6)
    off_diag = grad_eye - jnp.diag(jnp.diag(grad_eye))
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-7, rtol=1e-7)

    grad_zeros = jax.grad(lambda m: jnp.sum(safe_sqrt_sym(m)))(zeros)
    assert jnp.all(jnp.isfinite(grad_zeros))


def test_safe_mask_with_sqrt_is_gradient_safe():
    mask = jnp.array([True, False, True, False])
    x = jnp.array([4.0, 0.0, 9.0, 0.0], dtype=jnp.float32)

    result = safe_mask(mask, jnp.sqrt, x, placeholder=0.0)
    np.testing.assert_allclose(result, jnp.array([2.0, 0.0, 3.0, 0.0], dtype=jnp.float32), atol=1e-6)

    grad = jax.grad(lambda y: jnp.sum(safe_mask(mask, jnp.sqrt, y)))(x)
    assert jnp.all(jnp.isfinite(grad))
    np.testing.assert_allclose(grad[~mask], 0.0, atol=1e-7)


def test_diagonal_self_pairs():
    ff = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    result = diagonal_self_pairs(ff)
    np.testing.assert_allclose(result, jnp.array([35.0, 56.0], dtype=jnp.float32), atol=1e-6)

