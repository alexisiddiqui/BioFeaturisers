"""Tests for SAXS end-to-end predict wrapper."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from biotite.structure.io.pdb import PDBFile

from biofeaturisers.config import SAXSConfig
from biofeaturisers.saxs.debye import saxs_six_partials
from biofeaturisers.saxs.featurise import featurise
from biofeaturisers.saxs.foxs import saxs_combine
from biofeaturisers.saxs.forward import forward
from biofeaturisers.saxs.predict import predict


def _load_1ubq_fragment():
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "1ubq_A_1_15.pdb"
    pdb = PDBFile.read(fixture_path)
    return pdb.get_structure(model=1)


def test_predict_matches_featurise_then_forward() -> None:
    atom_array = _load_1ubq_fragment()
    config = SAXSConfig(n_q=36, chunk_size=64, fit_c1_c2=False, c1=1.02, c2=1.1)
    coords = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))

    predicted = predict(atom_array=atom_array, config=config, coords=coords)
    features = featurise(atom_array=atom_array, config=config)
    direct = forward(coords=coords, features=features, config=config)
    np.testing.assert_allclose(np.asarray(predicted), np.asarray(direct), atol=1e-5)


def test_predict_trajectory_translation_invariance() -> None:
    atom_array = _load_1ubq_fragment()
    config = SAXSConfig(n_q=30, chunk_size=64, fit_c1_c2=False, c1=1.0, c2=0.0, batch_size=2)
    coords = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))
    translated = coords + jnp.asarray([25.0, -10.0, 4.0], dtype=jnp.float32)
    trajectory = jnp.stack([coords, translated], axis=0)

    single = predict(atom_array=atom_array, config=config, coords=coords)
    traj = predict(atom_array=atom_array, config=config, coords=trajectory)
    np.testing.assert_allclose(np.asarray(traj), np.asarray(single), atol=2e-1)


def test_predict_fits_c1_c2_from_synthetic_target() -> None:
    atom_array = _load_1ubq_fragment()
    config = SAXSConfig(
        n_q=28,
        chunk_size=64,
        fit_c1_c2=True,
        c1_range=(1.0, 1.1),
        c2_range=(1.0, 3.0),
        c1_steps=11,
        c2_steps=21,
    )
    coords = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))
    features = featurise(atom_array=atom_array, config=config)
    partials = saxs_six_partials(coords, features, chunk_size=config.chunk_size)

    c1_true, c2_true = 1.05, 2.0
    i_exp = 1.8 * saxs_combine(partials, c1=c1_true, c2=c2_true)
    sigma = jnp.ones_like(i_exp) * 0.05

    i_q, chi2, c1_fit, c2_fit = predict(
        atom_array=atom_array,
        config=config,
        coords=coords,
        i_exp=i_exp,
        sigma=sigma,
    )
    np.testing.assert_allclose(np.asarray(i_q), np.asarray(saxs_combine(partials, c1_fit, c2_fit)), atol=1e-6)
    np.testing.assert_allclose(c1_fit, c1_true, atol=1e-6)
    np.testing.assert_allclose(c2_fit, c2_true, atol=1e-6)
    assert chi2 < 1e-5
