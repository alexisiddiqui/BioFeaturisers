"""Tests for SAXS Debye six-partial kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.safe_math import safe_sinc
from biofeaturisers.saxs.debye import saxs_six_partials
from biofeaturisers.saxs.features import SAXSFeatures


def _make_features(simple_topology, simple_output_index, ff_vac, ff_excl, ff_water, q_values):
    n_sel = int(ff_vac.shape[0])
    return SAXSFeatures(
        topology=simple_topology,
        output_index=simple_output_index,
        atom_idx=jnp.arange(n_sel, dtype=jnp.int32),
        ff_vac=jnp.asarray(ff_vac, dtype=jnp.float32),
        ff_excl=jnp.asarray(ff_excl, dtype=jnp.float32),
        ff_water=jnp.asarray(ff_water, dtype=jnp.float32),
        solvent_acc=jnp.ones((n_sel,), dtype=jnp.float32),
        q_values=jnp.asarray(q_values, dtype=jnp.float32),
        chain_ids=np.asarray(["A"] * n_sel, dtype=str),
    )


def _dense_debye(coords: jax.Array, ff: jax.Array, q_values: jax.Array) -> jax.Array:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    qr = q_values[None, None, :] * dist[:, :, None]
    sinc_vals = safe_sinc(qr)
    return jnp.sum(ff[:, None, :] * ff[None, :, :] * sinc_vals, axis=(0, 1))


def test_two_atom_partial_matches_analytic(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)
    q_values = jnp.asarray([0.3], dtype=jnp.float32)
    ff_vac = jnp.asarray([[1.0], [1.0]], dtype=jnp.float32)
    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=jnp.zeros_like(ff_vac),
        ff_water=jnp.zeros_like(ff_vac),
        q_values=q_values,
    )

    partials = saxs_six_partials(coords=coords, features=features, chunk_size=2)
    expected = 2.0 + 2.0 * float(np.sin(0.3 * 5.0) / (0.3 * 5.0))
    np.testing.assert_allclose(np.asarray(partials[0, 0]), expected, atol=1e-5)


def test_chunk_size_independence_and_dense_match(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.2, 0.0],
            [2.5, -0.1, 0.3],
            [3.1, 1.0, -0.4],
        ],
        dtype=jnp.float32,
    )
    q_values = jnp.linspace(0.01, 0.5, 9, dtype=jnp.float32)
    ff_vac = jnp.asarray(
        [
            [1.0, 1.2, 1.4, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5],
            [0.9, 1.0, 1.1, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4],
            [1.3, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6],
            [0.8, 0.85, 0.9, 0.88, 0.82, 0.76, 0.71, 0.66, 0.61],
        ],
        dtype=jnp.float32,
    )
    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=jnp.zeros_like(ff_vac),
        ff_water=jnp.zeros_like(ff_vac),
        q_values=q_values,
    )

    p2 = saxs_six_partials(coords=coords, features=features, chunk_size=2)[0]
    p4 = saxs_six_partials(coords=coords, features=features, chunk_size=4)[0]
    dense = _dense_debye(coords=coords, ff=ff_vac, q_values=q_values)

    np.testing.assert_allclose(np.asarray(p2), np.asarray(p4), atol=1e-5)
    np.testing.assert_allclose(np.asarray(p2), np.asarray(dense), atol=1e-4)


def test_zero_ff_padding_atoms_contribute_nothing(simple_topology, simple_output_index) -> None:
    coords_base = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.5, -0.3]], dtype=jnp.float32)
    q_values = jnp.linspace(0.02, 0.4, 8, dtype=jnp.float32)
    ff_base = jnp.asarray([[1.0] * 8, [0.75] * 8], dtype=jnp.float32)

    features_base = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_base,
        ff_excl=jnp.zeros_like(ff_base),
        ff_water=jnp.zeros_like(ff_base),
        q_values=q_values,
    )
    partials_base = saxs_six_partials(coords=coords_base, features=features_base, chunk_size=2)

    coords_pad = jnp.concatenate(
        [coords_base, jnp.asarray([[5.0, 5.0, 5.0], [-4.0, 1.0, 2.0]], dtype=jnp.float32)],
        axis=0,
    )
    ff_pad = jnp.concatenate([ff_base, jnp.zeros((2, 8), dtype=jnp.float32)], axis=0)
    features_pad = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_pad,
        ff_excl=jnp.zeros_like(ff_pad),
        ff_water=jnp.zeros_like(ff_pad),
        q_values=q_values,
    )
    partials_pad = saxs_six_partials(coords=coords_pad, features=features_pad, chunk_size=2)
    np.testing.assert_allclose(np.asarray(partials_pad), np.asarray(partials_base), atol=1e-6)


def test_gradient_through_six_partials_is_finite(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.2, -0.1],
            [2.8, -0.5, 0.6],
        ],
        dtype=jnp.float32,
    )
    q_values = jnp.linspace(0.01, 0.3, 7, dtype=jnp.float32)
    ff_vac = jnp.asarray(
        [
            [1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55],
            [0.95, 0.85, 0.75, 0.7, 0.62, 0.58, 0.5],
            [0.8, 0.75, 0.7, 0.64, 0.59, 0.55, 0.52],
        ],
        dtype=jnp.float32,
    )
    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=jnp.zeros_like(ff_vac),
        ff_water=jnp.zeros_like(ff_vac),
        q_values=q_values,
    )
    grad = jax.grad(lambda c: jnp.sum(saxs_six_partials(c, features, chunk_size=2)))(coords)
    assert np.isfinite(np.asarray(grad)).all()

