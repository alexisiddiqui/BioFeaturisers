"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

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


@pytest.fixture
def pdb_5pti_frame0() -> AtomArray:
    """Load 5PTI preprocessed structure (first frame, no solvent, D->H).
    
    This is a global fixture for the 5PTI structure used across regression
    and validation tests. The structure has been preprocessed to:
    - Remove solvent molecules (HOH, WAT, ions)
    - Keep all hydrogens
    - Normalize deuterium (D) to hydrogen (H) for form factor compatibility
    
    Returns:
        AtomArray: The structure with coordinates and topology (1,104 atoms).
    """
    fixture_path = Path(__file__).resolve().parent / "regression" / "FOXS" / "fixtures" / "5PTI_frame0.pdb"
    pdb = PDBFile.read(fixture_path)
    return pdb.get_structure(model=1)
