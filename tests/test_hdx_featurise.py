import numpy as np
import biotite.structure as struc

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx.featurise import build_exclusion_mask, featurise
from biofeaturisers.hdx.forward import hdx_forward


def _make_two_residue_atom_array_without_h() -> struc.AtomArray:
    atom = struc.AtomArray(8)
    atom.coord = np.array(
        [
            [0.0, 0.0, 0.0],  # A:1 N
            [1.5, 0.0, 0.0],  # A:1 CA
            [2.7, 0.0, 0.0],  # A:1 C
            [3.7, 0.0, 0.0],  # A:1 O
            [4.1, 0.2, 0.0],  # A:2 N
            [5.4, 0.2, 0.0],  # A:2 CA
            [6.6, 0.2, 0.0],  # A:2 C
            [7.6, 0.2, 0.0],  # A:2 O
        ],
        dtype=np.float32,
    )
    atom.atom_name = np.array(["N", "CA", "C", "O", "N", "CA", "C", "O"])
    atom.res_name = np.array(["ALA", "ALA", "ALA", "ALA", "GLY", "GLY", "GLY", "GLY"])
    atom.res_id = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    atom.chain_id = np.array(["A"] * 8)
    atom.element = np.array(["N", "C", "C", "O", "N", "C", "C", "O"])
    atom.hetero = np.zeros(8, dtype=bool)
    return atom


def test_build_exclusion_mask_cross_and_intra_chain_logic():
    probe_resids = np.array([1, 2, 1, 2], dtype=np.int32)
    probe_chains = np.array(["A", "A", "B", "B"], dtype=str)
    env_resids = np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)
    env_chains = np.array(["A", "A", "A", "B", "B", "B"], dtype=str)

    mask = build_exclusion_mask(probe_resids, probe_chains, env_resids, env_chains, min_sep=2)
    expected = np.array(
        [
            [0, 0, 0, 1, 1, 1],  # A:1
            [0, 0, 0, 1, 1, 1],  # A:2
            [1, 1, 1, 0, 0, 0],  # B:1
            [1, 1, 1, 0, 0, 0],  # B:2
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(mask, expected)

    intrachain_only = build_exclusion_mask(
        probe_resids, probe_chains, env_resids, env_chains, min_sep=2, intrachain_only=True
    )
    np.testing.assert_array_equal(intrachain_only, np.zeros_like(expected))


def test_featurise_generates_virtual_amide_h_when_missing():
    atom_array = _make_two_residue_atom_array_without_h()
    config = HDXConfig(seq_sep_min=0)
    features = featurise(atom_array, config=config)
    coords = atom_array.coord.astype(np.float32)
    result = hdx_forward(coords, features, config)

    assert features.res_keys.tolist() == ["A:2"]
    np.testing.assert_array_equal(features.amide_has_observed_H, np.array([False]))
    assert int(features.amide_C_prev_idx[0]) == 2
    assert result["ln_Pf"].shape == (1,)
    assert np.isfinite(np.asarray(result["ln_Pf"])).all()
