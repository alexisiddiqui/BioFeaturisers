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


def test_switch_midpoint_identity():
    for r0 in [2.4, 6.5, 10.0]:
        np.testing.assert_allclose(sigmoid_switch(jnp.float32(r0), r0, b=10.0), 0.5, atol=1e-6)
        np.testing.assert_allclose(tanh_switch(jnp.float32(r0), r0, k=5.0), 0.5, atol=1e-6)
        np.testing.assert_allclose(rational_switch(jnp.float32(r0), r0, n=6, m=12), 0.5, atol=1e-6)


def test_sigmoid_tanh_equivalence():
    r = jnp.linspace(0.0, 15.0, 200, dtype=jnp.float32)
    for b in [3.0, 5.0, 10.0, 20.0]:
        s = sigmoid_switch(r, r0=6.5, b=b)
        t = tanh_switch(r, r0=6.5, k=b / 2.0)
        np.testing.assert_allclose(s, t, atol=1e-6, rtol=1e-6)


def test_sigmoid_monotonicity_and_limits():
    r = jnp.linspace(0.0, 20.0, 500, dtype=jnp.float32)
    s = sigmoid_switch(r, r0=6.5, b=10.0)
    assert np.all(np.diff(np.asarray(s)) <= 1e-7)

    np.testing.assert_allclose(sigmoid_switch(jnp.float32(0.0), 6.5, 10.0), 1.0, atol=1e-6)
    np.testing.assert_allclose(sigmoid_switch(jnp.float32(20.0), 6.5, 10.0), 0.0, atol=1e-6)


def test_sigmoid_gradient_is_finite_and_peaks_near_midpoint():
    r = jnp.linspace(0.0, 15.0, 200, dtype=jnp.float32)
    grads = jax.vmap(jax.grad(lambda x: sigmoid_switch(x, 6.5, 10.0)))(r)
    assert jnp.all(jnp.isfinite(grads))

    peak_r = float(r[jnp.argmax(jnp.abs(grads))])
    assert abs(peak_r - 6.5) < 0.2


def test_rational_switch_handles_singularity():
    value = rational_switch(jnp.float32(6.5), 6.5, n=6, m=12)
    np.testing.assert_allclose(value, 0.5, atol=1e-6)

    grad = jax.grad(lambda x: rational_switch(x, 6.5))(jnp.float32(6.5))
    assert jnp.isfinite(grad)


def test_apply_switch_grid_shape_and_consistency():
    key = jax.random.PRNGKey(0)
    dist = jnp.abs(jax.random.normal(key, (10, 50), dtype=jnp.float32))
    mask = jnp.ones((10, 50), dtype=jnp.float32)
    r0s = jnp.array([5.0, 6.5, 8.0], dtype=jnp.float32)
    bs = jnp.array([5.0, 10.0], dtype=jnp.float32)

    result = apply_switch_grid(dist, mask, r0s, bs)
    assert result.shape == (3, 2, 10)

    direct = jnp.sum(sigmoid_switch(dist, r0=6.5, b=10.0) * mask, axis=-1)
    np.testing.assert_allclose(result[1, 1, :], direct, atol=1e-6, rtol=1e-6)


def test_bv_contact_counts_matches_direct_formula_and_masks():
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],  # amide N
            [0.0, 1.0, 0.0],  # amide H
            [3.0, 4.0, 0.0],  # heavy atom for Nc (distance 5 from N)
            [0.0, 1.5, 0.0],  # backbone O for Nh (distance 0.5 from H)
        ],
        dtype=jnp.float32,
    )

    amide_n_idx = jnp.array([0], dtype=jnp.int32)
    amide_h_idx = jnp.array([1], dtype=jnp.int32)
    heavy_idx = jnp.array([2], dtype=jnp.int32)
    o_idx = jnp.array([3], dtype=jnp.int32)

    mask_c = jnp.array([[1.0]], dtype=jnp.float32)
    mask_h = jnp.array([[1.0]], dtype=jnp.float32)

    nc, nh = bv_contact_counts(
        coords=coords,
        amide_N_idx=amide_n_idx,
        amide_H_idx=amide_h_idx,
        heavy_idx=heavy_idx,
        backbone_O_idx=o_idx,
        excl_mask_c=mask_c,
        excl_mask_h=mask_h,
        x_c=6.5,
        x_h=2.4,
        b=10.0,
    )

    expected_nc = sigmoid_switch(jnp.float32(5.0), 6.5, 10.0)
    expected_nh = sigmoid_switch(jnp.float32(0.5), 2.4, 10.0)
    np.testing.assert_allclose(nc, expected_nc[None], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(nh, expected_nh[None], atol=1e-6, rtol=1e-6)

    nc_zero, nh_zero = bv_contact_counts(
        coords=coords,
        amide_N_idx=amide_n_idx,
        amide_H_idx=amide_h_idx,
        heavy_idx=heavy_idx,
        backbone_O_idx=o_idx,
        excl_mask_c=jnp.zeros_like(mask_c),
        excl_mask_h=jnp.zeros_like(mask_h),
        x_c=6.5,
        x_h=2.4,
        b=10.0,
    )
    np.testing.assert_allclose(nc_zero, 0.0, atol=1e-7)
    np.testing.assert_allclose(nh_zero, 0.0, atol=1e-7)

