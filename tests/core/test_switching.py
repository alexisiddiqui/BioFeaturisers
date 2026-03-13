"""Tests for switching kernels and BV contact-count logic."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.switching import (
    apply_switch_grid,
    bv_contact_counts,
    rational_switch,
    sigmoid_switch,
    tanh_switch,
)


def test_switch_midpoint_identity() -> None:
    for r0 in [2.4, 6.5, 10.0]:
        assert np.isclose(float(sigmoid_switch(r0, r0, b=10.0)), 0.5, atol=1e-6)
        assert np.isclose(float(tanh_switch(r0, r0, k=5.0)), 0.5, atol=1e-6)
        assert np.isclose(float(rational_switch(r0, r0, n=6, m=12)), 0.5, atol=1e-6)


def test_sigmoid_and_tanh_switch_equivalence() -> None:
    r = jnp.linspace(0.0, 15.0, 200, dtype=jnp.float32)
    for b in [3.0, 5.0, 10.0, 20.0]:
        s = sigmoid_switch(r, r0=6.5, b=b)
        t = tanh_switch(r, r0=6.5, k=b / 2.0)
        np.testing.assert_allclose(np.asarray(s), np.asarray(t), atol=1e-6, rtol=1e-6)


def test_sigmoid_monotonic_limits_and_gradient_finite() -> None:
    r = jnp.linspace(0.0, 20.0, 500, dtype=jnp.float32)
    s = np.asarray(sigmoid_switch(r, r0=6.5, b=10.0))
    assert np.all(np.diff(s) <= 1e-7)
    assert np.isclose(s[0], 1.0, atol=1e-6)
    assert np.isclose(s[-1], 0.0, atol=1e-6)

    grad_vals = jax.vmap(jax.grad(lambda rv: sigmoid_switch(rv, 6.5, 10.0)))(r)
    grad_np = np.asarray(grad_vals)
    assert np.isfinite(grad_np).all()
    peak_r = float(np.asarray(r)[np.argmax(np.abs(grad_np))])
    assert abs(peak_r - 6.5) < 0.2


def test_rational_switch_handles_singularity() -> None:
    val = float(rational_switch(6.5, 6.5, n=6, m=12))
    grad = float(jax.grad(lambda rv: rational_switch(rv, 6.5))(jnp.float32(6.5)))
    assert np.isclose(val, 0.5, atol=1e-6)
    assert np.isfinite(grad)


def test_apply_switch_grid_shape_and_consistency() -> None:
    key = jax.random.PRNGKey(0)
    dist = jnp.abs(jax.random.normal(key, (10, 50), dtype=jnp.float32))
    mask = jnp.ones((10, 50), dtype=jnp.float32)
    r0s = jnp.asarray([5.0, 6.5, 8.0], dtype=jnp.float32)
    bs = jnp.asarray([5.0, 10.0], dtype=jnp.float32)

    grid = apply_switch_grid(dist, mask, r0s, bs)
    assert grid.shape == (3, 2, 10)

    direct = jnp.sum(sigmoid_switch(dist, 6.5, 10.0) * mask, axis=-1)
    np.testing.assert_allclose(np.asarray(grid[1, 1, :]), np.asarray(direct), atol=1e-5, rtol=1e-6)


def test_bv_contact_counts_matches_manual_single_probe() -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],  # amide N
            [0.0, 1.0, 0.0],  # amide H
            [1.0, 0.0, 0.0],  # heavy atom
            [0.0, 2.0, 0.0],  # backbone O
        ],
        dtype=jnp.float32,
    )
    amide_n = jnp.asarray([0], dtype=jnp.int32)
    amide_h = jnp.asarray([1], dtype=jnp.int32)
    heavy = jnp.asarray([2], dtype=jnp.int32)
    backbone_o = jnp.asarray([3], dtype=jnp.int32)
    excl_c = jnp.asarray([[1.0]], dtype=jnp.float32)
    excl_h = jnp.asarray([[1.0]], dtype=jnp.float32)

    nc, nh = bv_contact_counts(
        coords,
        amide_n,
        amide_h,
        heavy,
        backbone_o,
        excl_c,
        excl_h,
        x_c=6.5,
        x_h=2.4,
        b=10.0,
    )

    expected_nc = float(sigmoid_switch(jnp.float32(1.0), 6.5, 10.0))
    expected_nh = float(sigmoid_switch(jnp.float32(1.0), 2.4, 10.0))
    assert np.isclose(float(nc[0]), expected_nc, atol=1e-6)
    assert np.isclose(float(nh[0]), expected_nh, atol=1e-6)


def test_bv_contact_counts_respects_exclusion_masks() -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    idx1 = jnp.asarray([0], dtype=jnp.int32)
    idx2 = jnp.asarray([1], dtype=jnp.int32)
    idx3 = jnp.asarray([2], dtype=jnp.int32)
    idx4 = jnp.asarray([3], dtype=jnp.int32)
    zero = jnp.asarray([[0.0]], dtype=jnp.float32)

    nc, nh = bv_contact_counts(coords, idx1, idx2, idx3, idx4, zero, zero)
    np.testing.assert_allclose(np.asarray(nc), np.asarray([0.0], dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(np.asarray(nh), np.asarray([0.0], dtype=np.float32), atol=1e-7)
