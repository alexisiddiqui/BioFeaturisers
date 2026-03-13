"""Tests for HDX forward kernels."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from biotite.structure.io.pdb import PDBFile

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx.featurise import featurise
from biofeaturisers.hdx.forward import bucket_size, hdx_forward


def _load_1ubq_fragment():
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "1ubq_A_1_15.pdb"
    pdb = PDBFile.read(fixture_path)
    return pdb.get_structure(model=1)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    safe_norm = np.where(norm > 1e-8, norm, 1.0)
    return values / safe_norm


def _hard_cutoff_ln_pf(coords: np.ndarray, features, config: HDXConfig) -> np.ndarray:
    amide_n_idx = np.asarray(features.amide_N_idx, dtype=np.int32)
    amide_h_idx = np.asarray(features.amide_H_idx, dtype=np.int32)
    amide_ca_idx = np.asarray(features.amide_CA_idx, dtype=np.int32)
    amide_prev_c_idx = np.asarray(features.amide_prev_C_idx, dtype=np.int32)
    heavy_idx = np.asarray(features.heavy_atom_idx, dtype=np.int32)
    backbone_o_idx = np.asarray(features.backbone_O_idx, dtype=np.int32)
    excl_mask_c = np.asarray(features.excl_mask_c, dtype=np.float32)
    excl_mask_h = np.asarray(features.excl_mask_h, dtype=np.float32)

    amide_n = coords[amide_n_idx]
    amide_ca = coords[amide_ca_idx]
    amide_prev_c = coords[amide_prev_c_idx]
    amide_h_synth = amide_n + 1.01 * _normalize_rows(
        _normalize_rows(amide_ca - amide_n) + _normalize_rows(amide_prev_c - amide_n)
    )

    safe_h_idx = np.where(amide_h_idx >= 0, amide_h_idx, 0)
    amide_h_obs = coords[safe_h_idx]
    amide_h = np.where(amide_h_idx[:, None] >= 0, amide_h_obs, amide_h_synth)

    heavy = coords[heavy_idx]
    backbone_o = coords[backbone_o_idx]
    dist_c = np.linalg.norm(amide_n[:, None, :] - heavy[None, :, :], axis=-1)
    dist_h = np.linalg.norm(amide_h[:, None, :] - backbone_o[None, :, :], axis=-1)

    nc = np.sum((dist_c <= config.cutoff_c).astype(np.float32) * excl_mask_c, axis=-1)
    nh = np.sum((dist_h <= config.cutoff_h).astype(np.float32) * excl_mask_h, axis=-1)
    return config.beta_0 + config.beta_c * nc + config.beta_h * nh


def test_hdx_forward_1ubq_hard_bv_correlation() -> None:
    atom_array = _load_1ubq_fragment()
    config = HDXConfig(steepness_c=50.0, steepness_h=50.0)
    features = featurise(atom_array, config=config)
    coords = jnp.asarray(atom_array.coord, dtype=jnp.float32)

    result = hdx_forward(coords, features, config=config)
    soft_ln_pf = np.asarray(result["ln_Pf"])
    hard_ln_pf = _hard_cutoff_ln_pf(np.asarray(coords), features, config=config)

    assert soft_ln_pf.shape == hard_ln_pf.shape
    assert soft_ln_pf.shape[0] >= 8  # enough points for a meaningful correlation gate

    r = float(np.corrcoef(soft_ln_pf, hard_ln_pf)[0, 1])
    assert np.isfinite(r)
    assert r > 0.95


def test_hdx_forward_translation_invariance() -> None:
    atom_array = _load_1ubq_fragment()
    config = HDXConfig()
    features = featurise(atom_array, config=config)
    coords = jnp.asarray(atom_array.coord, dtype=jnp.float32)
    shifted = coords + jnp.asarray([75.0, -10.0, 22.5], dtype=jnp.float32)

    base = hdx_forward(coords, features, config=config)
    moved = hdx_forward(shifted, features, config=config)
    np.testing.assert_allclose(np.asarray(base["Nc"]), np.asarray(moved["Nc"]), atol=1e-3)
    np.testing.assert_allclose(np.asarray(base["Nh"]), np.asarray(moved["Nh"]), atol=1e-3)
    np.testing.assert_allclose(np.asarray(base["ln_Pf"]), np.asarray(moved["ln_Pf"]), atol=1e-3)


def test_bucket_size_is_power_of_two() -> None:
    for n_items in [1, 2, 3, 5, 9, 32, 33, 100, 1000]:
        bucket = bucket_size(n_items)
        assert bucket >= n_items
        assert bucket & (bucket - 1) == 0
