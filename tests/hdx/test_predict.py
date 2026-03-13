"""Tests for HDX end-to-end predict wrapper."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from biotite.structure import AtomArray

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx.featurise import featurise
from biofeaturisers.hdx.forward import hdx_forward
from biofeaturisers.hdx.predict import predict
from biofeaturisers.hdx import hdxrate


def _make_three_residue_no_h_atom_array() -> AtomArray:
    atom_names = ["N", "CA", "C", "O"] * 3
    res_names = ["ALA"] * 4 + ["GLY"] * 4 + ["SER"] * 4
    res_ids = [1] * 4 + [2] * 4 + [3] * 4
    chain_ids = ["A"] * 12
    element = ["N", "C", "C", "O"] * 3
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


def test_predict_matches_featurise_then_forward() -> None:
    atom_array = _make_three_residue_no_h_atom_array()
    config = HDXConfig(seq_sep_min=0)
    result = predict(atom_array, config=config)

    features = featurise(atom_array, config=config)
    direct = hdx_forward(jnp.asarray(atom_array.coord, dtype=jnp.float32), features, config=config)

    np.testing.assert_allclose(np.asarray(result["Nc"]), np.asarray(direct["Nc"]))
    np.testing.assert_allclose(np.asarray(result["Nh"]), np.asarray(direct["Nh"]))
    np.testing.assert_allclose(np.asarray(result["ln_Pf"]), np.asarray(direct["ln_Pf"]))


def test_predict_trajectory_average_matches_single_frame_for_translation() -> None:
    atom_array = _make_three_residue_no_h_atom_array()
    config = HDXConfig(seq_sep_min=0)
    base_coords = jnp.asarray(atom_array.coord, dtype=jnp.float32)
    translated = base_coords + jnp.asarray([25.0, -10.0, 4.0], dtype=jnp.float32)
    trajectory = jnp.stack([base_coords, translated], axis=0)

    single = predict(atom_array, config=config, coords=base_coords)
    traj = predict(atom_array, config=config, coords=trajectory)

    np.testing.assert_allclose(np.asarray(traj["ln_Pf"]), np.asarray(single["ln_Pf"]), atol=1e-5)


def test_predict_returns_uptake_when_hdxrate_enabled(monkeypatch) -> None:
    atom_array = _make_three_residue_no_h_atom_array()

    def fake_k_int_from_sequence(sequence: str, temperature: float, pH: float) -> np.ndarray:
        rates = np.ones((len(sequence),), dtype=np.float32)
        rates[0] = 0.0  # emulate N-terminus rule from HDXrate
        return rates

    monkeypatch.setattr(hdxrate, "_load_hdxrate_api", lambda: fake_k_int_from_sequence)
    config = HDXConfig(use_hdxrate=True, timepoints=[1.0, 10.0], seq_sep_min=0)
    result = predict(atom_array, config=config)

    assert "uptake" in result
    assert result["uptake"].shape[1] == 2
    assert np.isfinite(np.asarray(result["uptake"])).all()

