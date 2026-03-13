"""Tests for OutputIndex selection semantics."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology
from tests.fixtures.biotite_builders import make_test_atom_array


@pytest.fixture
def topology() -> MinimalTopology:
    return MinimalTopology.from_biotite(make_test_atom_array())


def test_default_selection_excludes_hetatm(topology: MinimalTopology) -> None:
    output_index = OutputIndex.from_selection(topology)
    assert np.asarray(output_index.atom_mask).tolist() == [True, True, True, True, True, False]
    assert np.asarray(output_index.probe_mask).tolist() == [True, True, True, True, True, False]
    assert np.asarray(output_index.output_mask).tolist() == [True, True, False, True]
    assert isinstance(output_index.atom_mask, jax.Array)
    assert isinstance(output_index.atom_idx, jax.Array)
    chex.assert_shape(output_index.atom_mask, (6,))
    chex.assert_shape(output_index.atom_idx, (5,))
    assert output_index.atom_mask.dtype == jnp.bool_
    assert output_index.atom_idx.dtype == jnp.int32


def test_include_hetatm_in_environment_only(topology: MinimalTopology) -> None:
    output_index = OutputIndex.from_selection(topology, include_hetatm=True)
    assert np.asarray(output_index.atom_mask).tolist() == [True, True, True, True, True, True]
    assert np.asarray(output_index.probe_mask).tolist() == [True, True, True, True, True, False]


def test_chain_filtering(topology: MinimalTopology) -> None:
    output_index = OutputIndex.from_selection(topology, include_chains=["B"])
    assert np.asarray(output_index.atom_mask).tolist() == [False, False, False, False, True, False]
    assert np.asarray(output_index.output_res_idx).tolist() == [3]


def test_custom_atom_mask_override(topology: MinimalTopology) -> None:
    custom = np.asarray([False, False, True, False, False, False], dtype=bool)
    output_index = OutputIndex.from_selection(topology, custom_atom_mask=custom)
    assert np.asarray(output_index.atom_idx).tolist() == [2]
    assert np.asarray(output_index.probe_idx).tolist() == [2]
    assert np.asarray(output_index.output_res_idx).tolist() == [1]


def test_custom_mask_shape_validation(topology: MinimalTopology) -> None:
    with pytest.raises(ValueError):
        OutputIndex.from_selection(topology, custom_atom_mask=np.asarray([True, False]))
