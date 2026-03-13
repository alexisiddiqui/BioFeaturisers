"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology


@pytest.fixture
def simple_topology() -> MinimalTopology:
    """Small, deterministic topology for dataclass contract tests."""
    return MinimalTopology(
        atom_names=np.asarray(["N", "CA", "C1", "N"], dtype=str),
        res_names=np.asarray(["ALA", "ALA", "LIG", "GLY"], dtype=str),
        res_ids=jnp.asarray([1, 1, 101, 5], dtype=jnp.int32),
        chain_ids=np.asarray(["A", "A", "A", "B"], dtype=str),
        element=np.asarray(["N", "C", "C", "N"], dtype=str),
        is_hetatm=jnp.asarray([False, False, True, False], dtype=jnp.bool_),
        is_backbone=jnp.asarray([True, True, False, True], dtype=jnp.bool_),
        seg_ids=np.asarray(["", "", "", ""], dtype=str),
        res_unique_ids=np.asarray(["A:1", "A:101", "B:5"], dtype=str),
        res_can_exchange=jnp.asarray([False, False, False], dtype=jnp.bool_),
    )


@pytest.fixture
def simple_output_index(simple_topology: MinimalTopology) -> OutputIndex:
    return OutputIndex.from_selection(simple_topology)
