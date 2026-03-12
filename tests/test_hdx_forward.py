from pathlib import Path

import numpy as np
from biotite.structure.io.pdb import PDBFile, get_structure

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx.featurise import featurise
from biofeaturisers.hdx.forward import hdx_forward


def _analytic_amide_h(n_xyz: np.ndarray, ca_xyz: np.ndarray, c_prev_xyz: np.ndarray) -> np.ndarray:
    def _unit(v: np.ndarray) -> np.ndarray:
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

    v1 = _unit(ca_xyz - n_xyz)
    v2 = _unit(c_prev_xyz - n_xyz)
    bisector = _unit(v1 + v2)
    return n_xyz + 1.01 * bisector


def _hard_ln_pf(coords: np.ndarray, features, config: HDXConfig) -> np.ndarray:
    n_xyz = coords[features.amide_N_idx]
    observed_h = coords[features.amide_H_idx]
    virtual_h = _analytic_amide_h(
        n_xyz,
        coords[features.amide_CA_idx],
        coords[features.amide_C_prev_idx],
    )
    h_xyz = np.where(features.amide_has_observed_H[:, None], observed_h, virtual_h)

    heavy_xyz = coords[features.heavy_atom_idx]
    bb_o_xyz = coords[features.backbone_O_idx]
    dist_c = np.linalg.norm(n_xyz[:, None, :] - heavy_xyz[None, :, :], axis=-1)
    dist_h = np.linalg.norm(h_xyz[:, None, :] - bb_o_xyz[None, :, :], axis=-1)

    hard_nc = np.sum((dist_c < config.cutoff_c).astype(np.float32) * features.excl_mask_c, axis=-1)
    hard_nh = np.sum((dist_h < config.cutoff_h).astype(np.float32) * features.excl_mask_h, axis=-1)
    return config.beta_0 + config.beta_c * hard_nc + config.beta_h * hard_nh


def test_hdx_forward_1ubq_ln_pf_correlates_with_hard_bv_limits():
    pdb_path = Path(__file__).parent / "data" / "1UBQ.pdb"
    atom_array = get_structure(PDBFile.read(str(pdb_path)), model=1)
    coords = atom_array.coord.astype(np.float32)
    config = HDXConfig(
        beta_c=0.35,
        beta_h=2.0,
        beta_0=0.0,
        cutoff_c=6.5,
        cutoff_h=2.4,
        steepness=10.0,
        seq_sep_min=2,
    )

    features = featurise(atom_array, config=config)
    result = hdx_forward(coords, features, config)
    hard_ln_pf = _hard_ln_pf(coords, features, config)
    soft_ln_pf = np.asarray(result["ln_Pf"])

    assert result["Nc"].shape == hard_ln_pf.shape
    assert result["Nh"].shape == hard_ln_pf.shape
    assert np.isfinite(soft_ln_pf).all()

    pearson_r = float(np.corrcoef(soft_ln_pf, hard_ln_pf)[0, 1])
    assert pearson_r > 0.95


def test_hdx_forward_ignores_unreferenced_extra_atoms():
    pdb_path = Path(__file__).parent / "data" / "1UBQ.pdb"
    atom_array = get_structure(PDBFile.read(str(pdb_path)), model=1)
    coords = atom_array.coord.astype(np.float32)
    config = HDXConfig()

    features = featurise(atom_array, config=config)
    base = hdx_forward(coords, features, config)

    extra = np.zeros((17, 3), dtype=np.float32)
    padded_coords = np.concatenate([coords, extra], axis=0)
    padded = hdx_forward(padded_coords, features, config)

    np.testing.assert_allclose(np.asarray(base["Nc"]), np.asarray(padded["Nc"]), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(base["Nh"]), np.asarray(padded["Nh"]), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(base["ln_Pf"]), np.asarray(padded["ln_Pf"]), atol=1e-6, rtol=1e-6)
