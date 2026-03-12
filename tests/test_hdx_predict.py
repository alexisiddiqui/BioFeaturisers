import numpy as np
import biotite.structure as struc

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx.featurise import featurise
from biofeaturisers.hdx.forward import hdx_forward
from biofeaturisers.hdx.predict import predict


def _make_three_residue_atom_array() -> struc.AtomArray:
    atom = struc.AtomArray(15)
    atom.coord = np.array(
        [
            [0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.6, 0.0, 0.0], [3.6, 0.0, 0.0], [0.0, 1.0, 0.0],  # A:1
            [3.9, 0.2, 0.0], [5.2, 0.2, 0.0], [6.4, 0.2, 0.0], [7.4, 0.2, 0.0], [3.9, 1.2, 0.0],  # A:2
            [6.7, 0.4, 0.0], [8.0, 0.4, 0.0], [9.2, 0.4, 0.0], [10.2, 0.4, 0.0], [6.7, 1.4, 0.0], # A:3
        ],
        dtype=np.float32,
    )
    atom.atom_name = np.array(
        ["N", "CA", "C", "O", "H", "N", "CA", "C", "O", "H", "N", "CA", "C", "O", "H"]
    )
    atom.res_name = np.array(["ALA"] * 5 + ["GLY"] * 5 + ["LEU"] * 5)
    atom.res_id = np.array([1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)
    atom.chain_id = np.array(["A"] * 15)
    atom.element = np.array(["N", "C", "C", "O", "H"] * 3)
    atom.hetero = np.zeros(15, dtype=bool)
    return atom


def test_predict_matches_featurise_then_forward():
    atom_array = _make_three_residue_atom_array()
    config = HDXConfig(seq_sep_min=1)
    predicted = predict(atom_array, config=config)

    features = featurise(atom_array, config=config)
    direct = hdx_forward(atom_array.coord.astype(np.float32), features, config)

    np.testing.assert_allclose(np.asarray(predicted["Nc"]), np.asarray(direct["Nc"]), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(predicted["Nh"]), np.asarray(direct["Nh"]), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(predicted["ln_Pf"]), np.asarray(direct["ln_Pf"]), atol=1e-6, rtol=1e-6)
