"""Optional numerical stress tests for SAXS stability and gradient behavior."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from biofeaturisers.saxs.debye import saxs_six_partials
from biofeaturisers.saxs.features import SAXSFeatures
from tests.fixtures.numerical_helpers import make_linear_coords

_RUN_SLOW = os.getenv("BIOFEATURISERS_RUN_SLOW", "0") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN_SLOW,
    reason="Set BIOFEATURISERS_RUN_SLOW=1 to run stress/JIT optional suites.",
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


def test_very_large_coordinates_remain_finite(simple_topology, simple_output_index) -> None:
    n_atoms = 50
    coords = make_linear_coords(n_atoms, spacing=1000.0)
    q_values = jnp.linspace(0.01, 0.5, 12, dtype=jnp.float32)
    ff_vac = jnp.ones((n_atoms, q_values.shape[0]), dtype=jnp.float32)
    features = _make_features(
        simple_topology,
        simple_output_index,
        ff_vac=ff_vac,
        ff_excl=jnp.zeros_like(ff_vac),
        ff_water=jnp.zeros_like(ff_vac),
        q_values=q_values,
    )

    partials = saxs_six_partials(coords=coords, features=features, chunk_size=16)
    assert np.isfinite(np.asarray(partials)).all()


def test_very_small_distances_have_finite_gradients(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]], dtype=jnp.float32)
    q_values = jnp.asarray([0.05, 0.1, 0.2], dtype=jnp.float32)
    ff_vac = jnp.asarray([[1.0, 0.9, 0.8], [1.0, 0.9, 0.8]], dtype=jnp.float32)
    features = _make_features(
        simple_topology,
        simple_output_index,
        ff_vac=ff_vac,
        ff_excl=jnp.zeros_like(ff_vac),
        ff_water=jnp.zeros_like(ff_vac),
        q_values=q_values,
    )

    loss = lambda c: jnp.sum(saxs_six_partials(c, features, chunk_size=2))
    value = loss(coords)
    grad = jax.grad(loss)(coords)
    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(grad)).all()


def test_large_n_accumulation_consistency(simple_topology, simple_output_index) -> None:
    n_atoms = 5000
    q_values = jnp.linspace(0.02, 0.3, 6, dtype=jnp.float32)
    coords = make_linear_coords(n_atoms, spacing=1.1)
    ff_vac = jnp.ones((n_atoms, q_values.shape[0]), dtype=jnp.float32)
    features = _make_features(
        simple_topology,
        simple_output_index,
        ff_vac=ff_vac,
        ff_excl=jnp.zeros_like(ff_vac),
        ff_water=jnp.zeros_like(ff_vac),
        q_values=q_values,
    )

    p_128 = saxs_six_partials(coords=coords, features=features, chunk_size=128)[0]
    p_all = saxs_six_partials(coords=coords, features=features, chunk_size=n_atoms)[0]
    np.testing.assert_allclose(np.asarray(p_128), np.asarray(p_all), rtol=1e-3, atol=1e-3)


def test_gradient_checkpoint_matches_non_checkpoint(simple_topology, simple_output_index) -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.1, 0.0],
            [2.1, 0.8, -0.2],
            [3.0, 1.4, 0.3],
        ],
        dtype=jnp.float32,
    )
    q_values = jnp.linspace(0.01, 0.25, 7, dtype=jnp.float32)
    ff_vac = jnp.asarray(np.linspace(0.8, 1.3, 28, dtype=np.float32).reshape(4, 7))
    features = _make_features(
        simple_topology,
        simple_output_index,
        ff_vac=ff_vac,
        ff_excl=jnp.zeros_like(ff_vac),
        ff_water=jnp.zeros_like(ff_vac),
        q_values=q_values,
    )

    loss_no_cp = lambda c: jnp.sum(saxs_six_partials(c, features, chunk_size=2))
    loss_cp = lambda c: jnp.sum(jax.checkpoint(lambda z: saxs_six_partials(z, features, chunk_size=2))(c))

    g1 = jax.grad(loss_no_cp)(coords)
    g2 = jax.grad(loss_cp)(coords)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), atol=1e-5, rtol=1e-5)

