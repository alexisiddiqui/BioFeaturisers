"""Tests for shared numerical test helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tests.fixtures.numerical_helpers import (
    assert_directional_gradient_close,
    dense_debye_reference,
    make_linear_coords,
)


def test_make_linear_coords_spacing() -> None:
    coords = make_linear_coords(4, spacing=2.0)
    np.testing.assert_allclose(
        np.asarray(coords),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_dense_debye_reference_two_atoms() -> None:
    coords = jnp.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)
    ff = jnp.asarray([[1.0], [1.0]], dtype=jnp.float32)
    q = jnp.asarray([0.3], dtype=jnp.float32)
    out = dense_debye_reference(coords=coords, ff=ff, q_values=q)
    expected = 2.0 + 2.0 * float(np.sin(1.5) / 1.5)
    np.testing.assert_allclose(np.asarray(out), np.asarray([expected], dtype=np.float32), atol=1e-6)


def test_assert_directional_gradient_close_quadratic() -> None:
    x = jnp.asarray([[1.0, -2.0, 0.5], [0.2, -0.1, 0.3]], dtype=jnp.float32)
    assert_directional_gradient_close(lambda arr: jnp.sum(arr * arr), x, n_dirs=3, seed=7)

