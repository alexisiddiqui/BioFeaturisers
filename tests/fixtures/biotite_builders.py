"""Biotite builders for deterministic unit tests."""

from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray


def make_test_atom_array() -> AtomArray:
    """Build a small multi-chain AtomArray with one hetero residue."""
    array = AtomArray(6)
    array.atom_name = np.asarray(["N", "CA", "N", "CA", "N", "C1"], dtype="U4")
    array.res_name = np.asarray(["ALA", "ALA", "PRO", "PRO", "GLY", "LIG"], dtype="U4")
    array.res_id = np.asarray([1, 1, 2, 2, 5, 101], dtype=np.int32)
    array.chain_id = np.asarray(["A", "A", "A", "A", "B", "A"], dtype="U2")
    array.element = np.asarray(["N", "C", "N", "C", "N", "C"], dtype="U2")
    array.hetero = np.asarray([False, False, False, False, False, True], dtype=bool)
    array.coord = np.zeros((6, 3), dtype=np.float32)
    return array

