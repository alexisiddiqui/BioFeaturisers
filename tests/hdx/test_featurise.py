"""Tests for HDX featurisation helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from biotite.structure import AtomArray

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx.featurise import build_exclusion_mask, featurise
from biofeaturisers.hdx.forward import hdx_forward


def _make_three_residue_no_h_atom_array() -> AtomArray:
    atom_names = ["N", "CA", "C", "O"] * 3
    res_names = ["ALA"] * 4 + ["GLY"] * 4 + ["SER"] * 4
    res_ids = [1] * 4 + [2] * 4 + [3] * 4
    chain_ids = ["A"] * 12
    element = ["N", "C", "C", "O"] * 3

    # Rough peptide geometry without explicit amide hydrogens.
    coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.4, 1.1, 0.0],
            [3.5, 1.1, 0.0],
            [3.8, 0.0, 0.0],
            [5.2, 0.0, 0.0],
            [6.2, 1.1, 0.0],
            [7.3, 1.1, 0.0],
            [7.6, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [10.0, 1.1, 0.0],
            [11.1, 1.1, 0.0],
        ],
        dtype=np.float32,
    )

    atom_array = AtomArray(len(atom_names))
    atom_array.atom_name = np.asarray(atom_names, dtype="U4")
    atom_array.res_name = np.asarray(res_names, dtype="U4")
    atom_array.res_id = np.asarray(res_ids, dtype=np.int32)
    atom_array.chain_id = np.asarray(chain_ids, dtype="U2")
    atom_array.element = np.asarray(element, dtype="U2")
    atom_array.hetero = np.zeros(len(atom_names), dtype=bool)
    atom_array.coord = coord
    return atom_array


def test_build_exclusion_mask_cross_chain_logic() -> None:
    mask = build_exclusion_mask(
        probe_resids=np.asarray([1, 2], dtype=np.int32),
        probe_chain_ids=np.asarray(["A", "A"], dtype=str),
        env_resids=np.asarray([1, 2, 1], dtype=np.int32),
        env_chain_ids=np.asarray(["A", "A", "B"], dtype=str),
        min_sep=1,
    )
    expected = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(mask, expected)


def test_build_exclusion_mask_intrachain_only_blocks_cross_chain() -> None:
    mask = build_exclusion_mask(
        probe_resids=np.asarray([1], dtype=np.int32),
        probe_chain_ids=np.asarray(["A"], dtype=str),
        env_resids=np.asarray([4, 7], dtype=np.int32),
        env_chain_ids=np.asarray(["A", "B"], dtype=str),
        min_sep=2,
        intrachain_only=True,
    )
    np.testing.assert_array_equal(mask, np.asarray([[1.0, 0.0]], dtype=np.float32))


def test_featurise_tracks_missing_amide_h_with_analytical_geometry() -> None:
    atom_array = _make_three_residue_no_h_atom_array()
    config = HDXConfig(seq_sep_min=0)
    features = featurise(atom_array, config=config)

    # Residue 1 is N-terminal and excluded, residues 2-3 are exchangeable.
    assert features.amide_N_idx.shape == (2,)
    np.testing.assert_array_equal(np.asarray(features.amide_H_idx), np.asarray([-1, -1]))
    assert features.amide_CA_idx is not None
    assert features.amide_prev_C_idx is not None
    assert np.all(np.asarray(features.amide_CA_idx) >= 0)
    assert np.all(np.asarray(features.amide_prev_C_idx) >= 0)

    result = hdx_forward(jnp.asarray(atom_array.coord, dtype=jnp.float32), features, config=config)
    assert np.isfinite(np.asarray(result["ln_Pf"])).all()

