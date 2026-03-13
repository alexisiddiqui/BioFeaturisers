"""Tests for SAXSFeatures dataclass contract and persistence."""

from __future__ import annotations

import numpy as np
import pytest

from biofeaturisers.saxs.features import SAXSFeatures


def test_saxs_features_save_load_roundtrip(
    tmp_path, simple_topology, simple_output_index
) -> None:
    atom_idx = np.asarray([0, 1, 3], dtype=np.int32)
    n_q = 4
    features = SAXSFeatures(
        topology=simple_topology,
        output_index=simple_output_index,
        atom_idx=atom_idx,
        ff_vac=np.ones((3, n_q), dtype=np.float32),
        ff_excl=np.ones((3, n_q), dtype=np.float32) * 2.0,
        ff_water=np.ones((3, n_q), dtype=np.float32) * 0.5,
        solvent_acc=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        q_values=np.linspace(0.01, 0.2, n_q, dtype=np.float32),
        chain_ids=np.asarray(["A", "A", "B"], dtype=str),
    )

    prefix = tmp_path / "saxs_case"
    features.save(str(prefix))
    loaded = SAXSFeatures.load(str(prefix))

    np.testing.assert_array_equal(loaded.atom_idx, features.atom_idx)
    np.testing.assert_array_equal(loaded.ff_vac, features.ff_vac)
    np.testing.assert_array_equal(loaded.q_values, features.q_values)
    np.testing.assert_array_equal(loaded.output_index.probe_idx, features.output_index.probe_idx)


def test_saxs_features_validate_shapes(simple_topology, simple_output_index) -> None:
    with pytest.raises(ValueError):
        SAXSFeatures(
            topology=simple_topology,
            output_index=simple_output_index,
            atom_idx=np.asarray([0, 1], dtype=np.int32),
            ff_vac=np.ones((1, 5), dtype=np.float32),
            ff_excl=np.ones((2, 5), dtype=np.float32),
            ff_water=np.ones((2, 5), dtype=np.float32),
            solvent_acc=np.asarray([0.1, 0.2], dtype=np.float32),
            q_values=np.linspace(0.01, 0.3, 5, dtype=np.float32),
            chain_ids=np.asarray(["A", "A"], dtype=str),
        )

