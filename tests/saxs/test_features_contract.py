"""Tests for SAXSFeatures dataclass contract and persistence."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from biofeaturisers.saxs.features import SAXSFeatures


def test_saxs_features_save_load_roundtrip(
    tmp_path, simple_topology, simple_output_index
) -> None:
    atom_idx = jnp.asarray([0, 1, 3], dtype=jnp.int32)
    n_q = 4
    features = SAXSFeatures(
        topology=simple_topology,
        output_index=simple_output_index,
        atom_idx=atom_idx,
        ff_vac=jnp.ones((3, n_q), dtype=jnp.float32),
        ff_excl=jnp.ones((3, n_q), dtype=jnp.float32) * 2.0,
        ff_water=jnp.ones((3, n_q), dtype=jnp.float32) * 0.5,
        solvent_acc=jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
        q_values=jnp.linspace(0.01, 0.2, n_q, dtype=jnp.float32),
        chain_ids=np.asarray(["A", "A", "B"], dtype=str),
    )

    prefix = tmp_path / "saxs_case"
    features.save(str(prefix))
    loaded = SAXSFeatures.load(str(prefix))

    np.testing.assert_array_equal(np.asarray(loaded.atom_idx), np.asarray(features.atom_idx))
    np.testing.assert_array_equal(np.asarray(loaded.ff_vac), np.asarray(features.ff_vac))
    np.testing.assert_array_equal(np.asarray(loaded.q_values), np.asarray(features.q_values))
    np.testing.assert_array_equal(
        np.asarray(loaded.output_index.probe_idx), np.asarray(features.output_index.probe_idx)
    )
    assert isinstance(loaded.atom_idx, jax.Array)
    assert isinstance(loaded.ff_vac, jax.Array)
    chex.assert_shape(loaded.atom_idx, (3,))
    chex.assert_shape(loaded.ff_vac, (3, 4))
    assert loaded.atom_idx.dtype == jnp.int32
    assert loaded.ff_vac.dtype == jnp.float32


def test_saxs_features_validate_shapes(simple_topology, simple_output_index) -> None:
    with pytest.raises(ValueError):
        SAXSFeatures(
            topology=simple_topology,
            output_index=simple_output_index,
            atom_idx=jnp.asarray([0, 1], dtype=jnp.int32),
            ff_vac=jnp.ones((1, 5), dtype=jnp.float32),
            ff_excl=jnp.ones((2, 5), dtype=jnp.float32),
            ff_water=jnp.ones((2, 5), dtype=jnp.float32),
            solvent_acc=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
            q_values=jnp.linspace(0.01, 0.3, 5, dtype=jnp.float32),
            chain_ids=np.asarray(["A", "A"], dtype=str),
        )
