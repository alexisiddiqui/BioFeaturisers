"""Tests for HDXFeatures dataclass contract and persistence."""

from __future__ import annotations

import numpy as np
import pytest

from biofeaturisers.hdx.features import HDXFeatures


def test_hdx_features_save_load_roundtrip(
    tmp_path, simple_topology, simple_output_index
) -> None:
    features = HDXFeatures(
        topology=simple_topology,
        output_index=simple_output_index,
        amide_N_idx=np.asarray([0], dtype=np.int32),
        amide_H_idx=np.asarray([0], dtype=np.int32),
        heavy_atom_idx=np.asarray([0, 1], dtype=np.int32),
        backbone_O_idx=np.asarray([1], dtype=np.int32),
        excl_mask_c=np.asarray([[1.0, 0.0]], dtype=np.float32),
        excl_mask_h=np.asarray([[1.0]], dtype=np.float32),
        res_keys=np.asarray(["A:1", "B:5"], dtype=str),
        res_names=np.asarray(["ALA", "GLY"], dtype=str),
        can_exchange=np.asarray([False, False], dtype=bool),
        kint=np.asarray([np.nan, np.nan], dtype=np.float32),
    )

    prefix = tmp_path / "hdx_case"
    features.save(str(prefix))
    loaded = HDXFeatures.load(str(prefix))

    np.testing.assert_array_equal(loaded.amide_N_idx, features.amide_N_idx)
    np.testing.assert_array_equal(loaded.excl_mask_c, features.excl_mask_c)
    np.testing.assert_array_equal(loaded.res_keys, features.res_keys)
    np.testing.assert_array_equal(loaded.output_index.atom_idx, features.output_index.atom_idx)
    assert loaded.kint is not None


def test_hdx_features_validate_lengths(simple_topology, simple_output_index) -> None:
    with pytest.raises(ValueError):
        HDXFeatures(
            topology=simple_topology,
            output_index=simple_output_index,
            amide_N_idx=np.asarray([0, 1], dtype=np.int32),
            amide_H_idx=np.asarray([0], dtype=np.int32),
            heavy_atom_idx=np.asarray([0, 1], dtype=np.int32),
            backbone_O_idx=np.asarray([1], dtype=np.int32),
            excl_mask_c=np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            excl_mask_h=np.asarray([[1.0], [1.0]], dtype=np.float32),
            res_keys=np.asarray(["A:1", "B:5"], dtype=str),
            res_names=np.asarray(["ALA", "GLY"], dtype=str),
            can_exchange=np.asarray([False, False], dtype=bool),
            kint=None,
        )

