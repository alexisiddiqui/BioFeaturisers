"""Tests for MinimalTopology contract and biotite extraction."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.topology import MinimalTopology
from tests.fixtures.biotite_builders import make_test_atom_array


def test_minimal_topology_from_biotite_extracts_expected_fields() -> None:
    atom_array = make_test_atom_array()
    topology = MinimalTopology.from_biotite(atom_array)

    assert topology.atom_names.shape == (6,)
    assert topology.res_unique_ids.tolist() == ["A:1", "A:2", "A:101", "B:5"]
    assert np.asarray(topology.is_backbone).tolist() == [True, True, True, True, True, False]
    assert np.asarray(topology.is_hetatm).tolist() == [False, False, False, False, False, True]
    assert isinstance(topology.res_ids, jax.Array)
    assert isinstance(topology.is_hetatm, jax.Array)
    assert isinstance(topology.res_can_exchange, jax.Array)
    chex.assert_shape(topology.res_ids, (6,))
    chex.assert_shape(topology.is_hetatm, (6,))
    assert topology.res_ids.dtype == jnp.int32
    assert topology.is_hetatm.dtype == jnp.bool_

    by_key = {key: idx for idx, key in enumerate(topology.res_unique_ids)}
    assert bool(topology.res_can_exchange[by_key["A:1"]]) is False  # chain A N-term
    assert bool(topology.res_can_exchange[by_key["A:2"]]) is False  # proline
    assert bool(topology.res_can_exchange[by_key["A:101"]]) is False  # hetero residue
    assert bool(topology.res_can_exchange[by_key["B:5"]]) is False  # chain B N-term


def test_minimal_topology_exchange_mask_applies() -> None:
    atom_array = make_test_atom_array()
    topology = MinimalTopology.from_biotite(atom_array, exchange_mask=["A:2"])
    by_key = {key: idx for idx, key in enumerate(topology.res_unique_ids)}
    assert bool(topology.res_can_exchange[by_key["A:2"]]) is False


def test_minimal_topology_json_roundtrip() -> None:
    atom_array = make_test_atom_array()
    original = MinimalTopology.from_biotite(atom_array)
    payload = original.to_json()
    loaded = MinimalTopology.from_json(payload)

    np.testing.assert_array_equal(loaded.atom_names, original.atom_names)
    np.testing.assert_array_equal(np.asarray(loaded.res_ids), np.asarray(original.res_ids))
    np.testing.assert_array_equal(loaded.chain_ids, original.chain_ids)
    np.testing.assert_array_equal(loaded.res_unique_ids, original.res_unique_ids)
    np.testing.assert_array_equal(
        np.asarray(loaded.res_can_exchange), np.asarray(original.res_can_exchange)
    )
