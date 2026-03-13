"""Tests for HDXFeatures dataclass contract and persistence."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from biofeaturisers.hdx.features import HDXFeatures


def test_hdx_features_save_load_roundtrip(
    tmp_path, simple_topology, simple_output_index
) -> None:
    features = HDXFeatures(
        topology=simple_topology,
        output_index=simple_output_index,
        amide_N_idx=jnp.asarray([0], dtype=jnp.int32),
        amide_H_idx=jnp.asarray([0], dtype=jnp.int32),
        heavy_atom_idx=jnp.asarray([0, 1], dtype=jnp.int32),
        backbone_O_idx=jnp.asarray([1], dtype=jnp.int32),
        excl_mask_c=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        excl_mask_h=jnp.asarray([[1.0]], dtype=jnp.float32),
        res_keys=np.asarray(["A:1", "B:5"], dtype=str),
        res_names=np.asarray(["ALA", "GLY"], dtype=str),
        can_exchange=jnp.asarray([False, False], dtype=jnp.bool_),
        kint=jnp.asarray([np.nan, np.nan], dtype=jnp.float32),
    )

    prefix = tmp_path / "hdx_case"
    features.save(str(prefix))
    loaded = HDXFeatures.load(str(prefix))

    np.testing.assert_array_equal(np.asarray(loaded.amide_N_idx), np.asarray(features.amide_N_idx))
    np.testing.assert_array_equal(np.asarray(loaded.excl_mask_c), np.asarray(features.excl_mask_c))
    np.testing.assert_array_equal(loaded.res_keys, features.res_keys)
    np.testing.assert_array_equal(
        np.asarray(loaded.output_index.atom_idx), np.asarray(features.output_index.atom_idx)
    )
    assert isinstance(loaded.amide_N_idx, jax.Array)
    assert isinstance(loaded.excl_mask_c, jax.Array)
    chex.assert_shape(loaded.amide_N_idx, (1,))
    chex.assert_shape(loaded.excl_mask_c, (1, 2))
    assert loaded.amide_N_idx.dtype == jnp.int32
    assert loaded.excl_mask_c.dtype == jnp.float32
    assert loaded.kint is not None


def test_hdx_features_validate_lengths(simple_topology, simple_output_index) -> None:
    with pytest.raises(ValueError):
        HDXFeatures(
            topology=simple_topology,
            output_index=simple_output_index,
            amide_N_idx=jnp.asarray([0, 1], dtype=jnp.int32),
            amide_H_idx=jnp.asarray([0], dtype=jnp.int32),
            heavy_atom_idx=jnp.asarray([0, 1], dtype=jnp.int32),
            backbone_O_idx=jnp.asarray([1], dtype=jnp.int32),
            excl_mask_c=jnp.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32),
            excl_mask_h=jnp.asarray([[1.0], [1.0]], dtype=jnp.float32),
            res_keys=np.asarray(["A:1", "B:5"], dtype=str),
            res_names=np.asarray(["ALA", "GLY"], dtype=str),
            can_exchange=jnp.asarray([False, False], dtype=jnp.bool_),
            kint=None,
        )
