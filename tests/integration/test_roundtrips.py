"""Integration tests for featurise/forward/predict and persistence round-trips."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from biofeaturisers.config import HDXConfig, SAXSConfig
from biofeaturisers.hdx.features import HDXFeatures
from biofeaturisers.hdx.featurise import featurise as hdx_featurise
from biofeaturisers.hdx.forward import hdx_forward
from biofeaturisers.hdx.predict import predict as hdx_predict
from biofeaturisers.hdx import hdxrate
from biofeaturisers.io.load import load_hdx_output, load_saxs_output
from biofeaturisers.io.save import save_hdx_output, save_saxs_output
from biofeaturisers.saxs.features import SAXSFeatures
from biofeaturisers.saxs.featurise import featurise as saxs_featurise
from biofeaturisers.saxs.forward import forward as saxs_forward
from biofeaturisers.saxs.predict import predict as saxs_predict


def _load_1ubq_fragment() -> AtomArray:
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "1ubq_A_1_15.pdb"
    pdb = PDBFile.read(fixture_path)
    return pdb.get_structure(model=1)


def _exchangeable_index(features) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    mask = np.asarray(features.can_exchange, dtype=bool)
    res_keys = np.asarray(features.res_keys, dtype=str)[mask]
    res_names = np.asarray(features.res_names, dtype=str)[mask]
    can_exchange = np.ones(res_keys.shape[0], dtype=bool)
    kint = (
        None
        if features.kint is None
        else np.asarray(features.kint, dtype=np.float32)[mask]
    )
    return res_keys, res_names, can_exchange, kint


def _make_two_chain_atom_array() -> AtomArray:
    atom_names = ["N", "CA", "C", "O"] * 6
    res_names = ["ALA"] * 4 + ["GLY"] * 4 + ["SER"] * 4 + ["ALA"] * 4 + ["GLY"] * 4 + ["SER"] * 4
    res_ids = [1] * 4 + [2] * 4 + [3] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    chain_ids = ["A"] * 12 + ["B"] * 12
    element = ["N", "C", "C", "O"] * 6

    coords: list[list[float]] = []
    for chain_offset in [0.0, 6.0]:
        for res_idx in range(3):
            x = float(res_idx * 3.8)
            y = float(chain_offset)
            coords.extend(
                [
                    [x + 0.0, y + 0.0, 0.0],
                    [x + 1.4, y + 0.1, 0.0],
                    [x + 2.4, y + 1.1, 0.1],
                    [x + 3.4, y + 1.0, -0.1],
                ]
            )

    atom_array = AtomArray(len(atom_names))
    atom_array.atom_name = np.asarray(atom_names, dtype="U4")
    atom_array.res_name = np.asarray(res_names, dtype="U4")
    atom_array.res_id = np.asarray(res_ids, dtype=np.int32)
    atom_array.chain_id = np.asarray(chain_ids, dtype="U2")
    atom_array.element = np.asarray(element, dtype="U2")
    atom_array.hetero = np.zeros(len(atom_names), dtype=bool)
    atom_array.coord = np.asarray(coords, dtype=np.float32)
    return atom_array


def test_hdx_save_load_forward_roundtrip(tmp_path) -> None:
    atom_array = _load_1ubq_fragment()
    config = HDXConfig(seq_sep_min=0)
    coords = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))

    features = hdx_featurise(atom_array, config=config)
    direct = hdx_forward(coords, features=features, config=config)

    prefix = tmp_path / "hdx_case"
    features.save(str(prefix))
    loaded = HDXFeatures.load(str(prefix))
    loaded_result = hdx_forward(coords, features=loaded, config=config)

    np.testing.assert_allclose(np.asarray(loaded_result["Nc"]), np.asarray(direct["Nc"]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(loaded_result["Nh"]), np.asarray(direct["Nh"]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(loaded_result["ln_Pf"]), np.asarray(direct["ln_Pf"]), atol=1e-6)

    res_keys, res_names, can_exchange, kint = _exchangeable_index(features)
    out_prefix = tmp_path / "hdx_out"
    save_hdx_output(
        str(out_prefix),
        nc=np.asarray(direct["Nc"], dtype=np.float32),
        nh=np.asarray(direct["Nh"], dtype=np.float32),
        ln_pf=np.asarray(direct["ln_Pf"], dtype=np.float32),
        res_keys=res_keys,
        res_names=res_names,
        can_exchange=can_exchange,
        kint=kint,
    )
    arrays, index = load_hdx_output(str(out_prefix))
    np.testing.assert_allclose(arrays["ln_Pf"], np.asarray(direct["ln_Pf"], dtype=np.float32), atol=1e-6)
    assert index["res_keys"].shape[0] == arrays["ln_Pf"].shape[0]


def test_saxs_save_load_forward_roundtrip(tmp_path) -> None:
    atom_array = _load_1ubq_fragment()
    config = SAXSConfig(n_q=24, chunk_size=64, fit_c1_c2=False, c1=1.0, c2=0.0)
    coords = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))

    features = saxs_featurise(atom_array, config=config)
    direct = saxs_forward(coords=coords, features=features, config=config)

    prefix = tmp_path / "saxs_case"
    features.save(str(prefix))
    loaded = SAXSFeatures.load(str(prefix))
    loaded_result = saxs_forward(coords=coords, features=loaded, config=config)
    np.testing.assert_allclose(np.asarray(loaded_result), np.asarray(direct), atol=1e-6)

    out_prefix = tmp_path / "saxs_out"
    save_saxs_output(
        str(out_prefix),
        i_q=np.asarray(direct, dtype=np.float32),
        q_values=np.asarray(features.q_values, dtype=np.float32),
        chain_ids=np.asarray(features.chain_ids, dtype=str),
        c1_used=float(config.c1),
        c2_used=float(config.c2),
    )
    arrays, index = load_saxs_output(str(out_prefix))
    np.testing.assert_allclose(arrays["I_q"], np.asarray(direct, dtype=np.float32), atol=1e-6)
    assert index["chain_ids"].shape[0] >= 1


def test_predict_matches_featurise_then_forward() -> None:
    atom_array = _load_1ubq_fragment()
    hdx_config = HDXConfig(seq_sep_min=0)
    saxs_config = SAXSConfig(n_q=20, chunk_size=64, fit_c1_c2=False, c1=1.0, c2=0.0)
    coords = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))

    hdx_pred = hdx_predict(atom_array=atom_array, config=hdx_config, coords=coords)
    hdx_features = hdx_featurise(atom_array=atom_array, config=hdx_config)
    hdx_direct = hdx_forward(coords, features=hdx_features, config=hdx_config)
    np.testing.assert_allclose(np.asarray(hdx_pred["ln_Pf"]), np.asarray(hdx_direct["ln_Pf"]), atol=1e-6)

    saxs_pred = saxs_predict(atom_array=atom_array, config=saxs_config, coords=coords)
    saxs_features = saxs_featurise(atom_array=atom_array, config=saxs_config)
    saxs_direct = saxs_forward(coords=coords, features=saxs_features, config=saxs_config)
    np.testing.assert_allclose(np.asarray(saxs_pred), np.asarray(saxs_direct), atol=1e-6)


def test_multichain_hdxrate_consistency(monkeypatch) -> None:
    atom_array = _make_two_chain_atom_array()

    def fake_k_int_from_sequence(sequence: str, temperature: float, pH: float) -> np.ndarray:
        values = np.linspace(0.0, float(len(sequence) - 1), num=len(sequence), dtype=np.float32)
        values[0] = 0.0
        return values

    monkeypatch.setattr(hdxrate, "_load_hdxrate_api", lambda: fake_k_int_from_sequence)
    config = HDXConfig(use_hdxrate=True, timepoints=[1.0], seq_sep_min=0)
    features = hdx_featurise(atom_array=atom_array, config=config)

    assert features.kint is not None
    res_keys = np.asarray(features.res_keys, dtype=str)
    a1 = int(np.where(res_keys == "A:1")[0][0])
    b1 = int(np.where(res_keys == "B:1")[0][0])
    a2 = int(np.where(res_keys == "A:2")[0][0])
    assert np.isnan(np.asarray(features.kint)[a1])
    assert np.isnan(np.asarray(features.kint)[b1])
    assert np.isfinite(np.asarray(features.kint)[a2])

