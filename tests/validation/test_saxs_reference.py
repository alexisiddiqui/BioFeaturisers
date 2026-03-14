"""Reference-profile regression test for SAXS."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from biotite.structure.io.pdb import PDBFile

from biofeaturisers.config import SAXSConfig
from biofeaturisers.saxs.predict import predict


def _load_1ubq_fragment():
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "1ubq_A_1_15.pdb"
    pdb = PDBFile.read(fixture_path)
    return pdb.get_structure(model=1)


def _load_reference_curve():
    ref_path = Path(__file__).resolve().parent.parent / "fixtures" / "1ubq_A_1_15_foxs_reference.dat"
    data = np.loadtxt(ref_path, comments="#")
    return data[:, 0], data[:, 1], data[:, 2]


def test_saxs_profile_matches_committed_reference_with_low_chi2() -> None:
    atom_array = _load_1ubq_fragment()
    q_ref, i_ref, sigma_ref = _load_reference_curve()

    config = SAXSConfig(
        q_min=float(q_ref[0]),
        q_max=float(q_ref[-1]),
        n_q=int(q_ref.shape[0]),
        chunk_size=32,
        fit_c1_c2=True,
        c1_range=(0.95, 1.12),
        c2_range=(0.0, 4.0),
        c1_steps=18,
        c2_steps=17,
    )
    i_q, chi2, c1_fit, c2_fit = predict(
        atom_array=atom_array,
        config=config,
        i_exp=jnp.asarray(i_ref, dtype=jnp.float32),
        sigma=jnp.asarray(sigma_ref, dtype=jnp.float32),
    )

    assert i_q.shape == (q_ref.shape[0],)
    assert np.isfinite(np.asarray(i_q)).all()
    assert np.isfinite(chi2)
    assert 0.90 <= c1_fit <= 1.20
    assert -0.1 <= c2_fit <= 4.5
    assert chi2 < 1.1

