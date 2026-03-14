"""Tests for FoXS recombination and forward wrappers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.safe_math import safe_sinc
from biofeaturisers.saxs.debye import saxs_six_partials
from biofeaturisers.saxs.features import SAXSFeatures
from biofeaturisers.saxs.foxs import saxs_combine, saxs_forward, saxs_trajectory


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
    return jnp.sum(ff[:, None, :] * ff[None, :, :] * safe_sinc(qr), axis=(0, 1))


def test_combine_polynomial_identity_and_gradients() -> None:
    partials = jnp.asarray([[10.0], [3.0], [1.0], [4.0], [2.0], [0.5]], dtype=jnp.float32)
    c1, c2 = 1.05, 2.0
    value = saxs_combine(partials, c1=c1, c2=c2)
    expected = 10.0 - c1 * 4.0 + (c1**2) * 3.0 + c2 * 2.0 - c1 * c2 * 0.5 + (c2**2) * 1.0
    np.testing.assert_allclose(np.asarray(value), np.asarray([expected], dtype=np.float32), atol=1e-6)

    d_dc1 = jax.grad(lambda x: saxs_combine(partials, c1=x, c2=c2).sum())(jnp.float32(c1))
    expected_dc1 = -4.0 + 2.0 * c1 * 3.0 - c2 * 0.5
    np.testing.assert_allclose(np.asarray(d_dc1), np.asarray(expected_dc1, dtype=np.float32), atol=1e-6)


def test_vacuum_only_reduces_to_debye(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.2, 0.0],
            [2.2, 1.0, -0.3],
            [3.0, 1.4, 0.2],
        ],
        dtype=jnp.float32,
    )
    q_values = jnp.linspace(0.01, 0.4, 12, dtype=jnp.float32)
    ff_vac = jnp.asarray(np.linspace(0.8, 1.2, 48, dtype=np.float32).reshape(4, 12))
    ff_zero = jnp.zeros_like(ff_vac)
    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=ff_zero,
        ff_water=ff_zero,
        q_values=q_values,
    )

    partials = saxs_six_partials(coords, features, chunk_size=2)
    combined = saxs_combine(partials, c1=0.0, c2=0.0)
    dense = _dense_debye(coords=coords, ff=ff_vac, q_values=q_values)
    forward = saxs_forward(coords=coords, features=features, c1=0.0, c2=0.0, chunk_size=2)

    np.testing.assert_allclose(np.asarray(combined), np.asarray(dense), atol=1e-4)
    np.testing.assert_allclose(np.asarray(forward), np.asarray(dense), atol=1e-4)


def test_trajectory_translation_invariance(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [2.0, 1.2, -0.2],
        ],
        dtype=jnp.float32,
    )
    shifted = coords + jnp.asarray([10.0, -4.0, 7.0], dtype=jnp.float32)
    trajectory = jnp.stack([coords, shifted], axis=0)
    q_values = jnp.linspace(0.02, 0.3, 10, dtype=jnp.float32)
    ff_vac = jnp.asarray(
        [
            [1.0] * 10,
            [0.9] * 10,
            [0.8] * 10,
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

    single = saxs_forward(coords, features, c1=0.0, c2=0.0, chunk_size=2)
    averaged = saxs_trajectory(
        trajectory=trajectory,
        features=features,
        c1=0.0,
        c2=0.0,
        batch_size=2,
        chunk_size=2,
    )
    np.testing.assert_allclose(np.asarray(single), np.asarray(averaged), atol=1e-5)

