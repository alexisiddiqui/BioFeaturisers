"""Tests for FoXS recombination and forward wrappers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.saxs.debye import saxs_six_partials
from biofeaturisers.saxs.features import SAXSFeatures
from biofeaturisers.saxs.foxs import saxs_combine, saxs_forward, saxs_trajectory
from tests.fixtures.numerical_helpers import (
    assert_directional_gradient_close,
    dense_debye_reference,
)


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


def test_combine_polynomial_identity_and_gradients() -> None:
    partials = jnp.asarray([[10.0], [3.0], [1.0], [4.0], [2.0], [0.5]], dtype=jnp.float32)
    c1, c2 = 1.05, 2.0
    value = saxs_combine(partials, c1=c1, c2=c2)
    expected = 10.0 - c1 * 4.0 + (c1**2) * 3.0 + c2 * 2.0 - c1 * c2 * 0.5 + (c2**2) * 1.0
    np.testing.assert_allclose(np.asarray(value), np.asarray([expected], dtype=np.float32), atol=1e-6)

    d_dc1 = jax.grad(lambda x: saxs_combine(partials, c1=x, c2=c2).sum())(jnp.float32(c1))
    expected_dc1 = -4.0 + 2.0 * c1 * 3.0 - c2 * 0.5
    np.testing.assert_allclose(np.asarray(d_dc1), np.asarray(expected_dc1, dtype=np.float32), atol=1e-6)

    d_dc2 = jax.grad(lambda x: saxs_combine(partials, c1=c1, c2=x).sum())(jnp.float32(c2))
    expected_dc2 = 2.0 - c1 * 0.5 + 2.0 * c2 * 1.0
    np.testing.assert_allclose(np.asarray(d_dc2), np.asarray(expected_dc2, dtype=np.float32), atol=1e-6)


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
    dense = dense_debye_reference(coords=coords, ff=ff_vac, q_values=q_values)
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


def test_six_partials_permutation_invariance(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [2.3, -0.4, 0.2],
            [3.0, 0.7, 0.3],
        ],
        dtype=jnp.float32,
    )
    q_values = jnp.linspace(0.02, 0.4, 8, dtype=jnp.float32)
    ff_vac = jnp.asarray(np.linspace(0.8, 1.2, 32, dtype=np.float32).reshape(4, 8))
    ff_excl = ff_vac * 0.3
    ff_water = ff_vac * 0.1

    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=ff_excl,
        ff_water=ff_water,
        q_values=q_values,
    )
    partials = saxs_six_partials(coords=coords, features=features, chunk_size=2)

    perm = np.asarray([2, 0, 3, 1], dtype=np.int32)
    features_perm = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac[perm],
        ff_excl=ff_excl[perm],
        ff_water=ff_water[perm],
        q_values=q_values,
    )
    partials_perm = saxs_six_partials(coords=coords[perm], features=features_perm, chunk_size=2)
    np.testing.assert_allclose(np.asarray(partials_perm), np.asarray(partials), atol=1e-5)


def test_six_partials_gradient_matches_fd_directions(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.1, -0.1],
            [2.1, 0.9, 0.2],
        ],
        dtype=jnp.float32,
    )
    q_values = jnp.linspace(0.01, 0.3, 6, dtype=jnp.float32)
    ff_vac = jnp.asarray(
        [
            [1.0, 0.95, 0.9, 0.82, 0.75, 0.7],
            [0.9, 0.88, 0.81, 0.76, 0.71, 0.66],
            [0.85, 0.81, 0.78, 0.74, 0.69, 0.65],
        ],
        dtype=jnp.float32,
    )
    ff_excl = ff_vac * 0.25
    ff_water = ff_vac * 0.08
    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=ff_excl,
        ff_water=ff_water,
        q_values=q_values,
    )

    coeff = jnp.linspace(0.8, 1.7, 6, dtype=jnp.float32)[:, None]
    loss = lambda c: jnp.sum(saxs_six_partials(c, features, chunk_size=2) * coeff)
    assert_directional_gradient_close(loss, coords, eps=2e-3, n_dirs=3, seed=11, rtol=2e-1, atol=2e-2)


def test_colocated_atoms_partial_decomposition_identity(simple_topology, simple_output_index) -> None:
    n_atoms = 3
    q_values = jnp.linspace(0.02, 0.2, 5, dtype=jnp.float32)
    coords = jnp.zeros((n_atoms, 3), dtype=jnp.float32)
    ff_vac = jnp.asarray(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [0.8, 0.7, 0.6, 0.5, 0.4],
            [0.6, 0.5, 0.4, 0.3, 0.2],
        ],
        dtype=jnp.float32,
    )
    ff_excl = ff_vac * 0.5
    ff_water = ff_vac * 0.2
    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=ff_excl,
        ff_water=ff_water,
        q_values=q_values,
    )
    partials = saxs_six_partials(coords=coords, features=features, chunk_size=2)
    partials_np = np.asarray(partials)

    sum_v = np.sum(np.asarray(ff_vac), axis=0)
    sum_e = np.sum(np.asarray(ff_excl), axis=0)
    sum_s = np.sum(np.asarray(ff_water), axis=0)
    np.testing.assert_allclose(partials_np[0], sum_v**2, atol=1e-5)
    np.testing.assert_allclose(partials_np[1], sum_e**2, atol=1e-5)
    np.testing.assert_allclose(partials_np[2], sum_s**2, atol=1e-5)
    np.testing.assert_allclose(partials_np[3], 2.0 * sum_v * sum_e, atol=1e-5)
    np.testing.assert_allclose(partials_np[4], 2.0 * sum_v * sum_s, atol=1e-5)
    np.testing.assert_allclose(partials_np[5], 2.0 * sum_e * sum_s, atol=1e-5)

    c1, c2 = 1.03, 1.7
    combined = np.asarray(saxs_combine(partials, c1=c1, c2=c2))
    ff_eff = np.asarray(ff_vac) - c1 * np.asarray(ff_excl) + c2 * np.asarray(ff_water)
    dense = np.asarray(dense_debye_reference(coords=coords, ff=jnp.asarray(ff_eff), q_values=q_values))
    np.testing.assert_allclose(combined, dense, atol=1e-5)
