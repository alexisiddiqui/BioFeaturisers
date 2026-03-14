"""Tests for SAXS featurisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from biotite.structure.io.pdb import PDBFile

from biofeaturisers.config import SAXSConfig
from biofeaturisers.saxs.featurise import featurise
from tests.fixtures.biotite_builders import make_test_atom_array


def _load_1ubq_fragment():
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "1ubq_A_1_15.pdb"
    pdb = PDBFile.read(fixture_path)
    return pdb.get_structure(model=1)


def test_featurise_shapes_and_ranges_for_real_fixture() -> None:
    atom_array = _load_1ubq_fragment()
    config = SAXSConfig(n_q=40, chunk_size=64)
    features = featurise(atom_array, config=config)

    n_sel = int(features.atom_idx.shape[0])
    assert n_sel > 0
    assert features.ff_vac.shape == (n_sel, 40)
    assert features.ff_excl.shape == (n_sel, 40)
    assert features.ff_water.shape == (n_sel, 40)
    assert features.solvent_acc.shape == (n_sel,)
    assert features.q_values.shape == (40,)
    assert np.isfinite(np.asarray(features.ff_vac)).all()
    assert np.isfinite(np.asarray(features.ff_excl)).all()
    assert np.isfinite(np.asarray(features.ff_water)).all()
    assert np.all(np.asarray(features.solvent_acc) >= 0.0)
    assert np.all(np.asarray(features.solvent_acc) <= 1.0)


def test_featurise_respects_chain_and_hetatm_filters() -> None:
    atom_array = make_test_atom_array()

    default_features = featurise(atom_array, config=SAXSConfig(n_q=20))
    default_idx = np.asarray(default_features.atom_idx, dtype=np.int32)
    assert 5 not in set(default_idx.tolist())  # hetero atom excluded by default

    chain_a_features = featurise(
        atom_array,
        config=SAXSConfig(n_q=20, include_chains=["A"]),
    )
    chain_a_idx = np.asarray(chain_a_features.atom_idx, dtype=np.int32)
    assert np.all(chain_a_idx < 5)  # chain B atom is index 4

    with_hetatm = featurise(
        atom_array,
        config=SAXSConfig(n_q=20, include_hetatm=True),
    )
    assert 5 in set(np.asarray(with_hetatm.atom_idx, dtype=np.int32).tolist())

