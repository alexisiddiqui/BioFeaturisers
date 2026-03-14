"""Gradient tests for the HDX ``hdx_forward`` wrapper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from biotite.structure import AtomArray

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx.featurise import featurise
from biofeaturisers.hdx.forward import hdx_forward
from tests.fixtures.numerical_helpers import assert_directional_gradient_close


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


def test_hdx_forward_ln_pf_gradient_matches_fd() -> None:
    atom_array = _make_three_residue_no_h_atom_array()
    config = HDXConfig(seq_sep_min=0, chunk_size=0, steepness_c=6.0, steepness_h=8.0)
    features = featurise(atom_array, config=config)
    coords = jnp.asarray(atom_array.coord, dtype=jnp.float32)

    loss = lambda c: jnp.sum(hdx_forward(c, features=features, config=config)["ln_Pf"])
    assert_directional_gradient_close(loss, coords, eps=1e-3, n_dirs=3, seed=31, rtol=2e-1, atol=2e-2)


def test_hdx_forward_chunked_and_dense_gradients_agree() -> None:
    atom_array = _make_three_residue_no_h_atom_array()
    dense_config = HDXConfig(seq_sep_min=0, chunk_size=0)
    chunked_config = HDXConfig(seq_sep_min=0, chunk_size=2)
    features = featurise(atom_array, config=dense_config)
    coords = jnp.asarray(atom_array.coord, dtype=jnp.float32)

    dense_loss = lambda c: jnp.sum(hdx_forward(c, features=features, config=dense_config)["ln_Pf"])
    chunked_loss = lambda c: jnp.sum(hdx_forward(c, features=features, config=chunked_config)["ln_Pf"])

    grad_dense = jax.grad(dense_loss)(coords)
    grad_chunked = jax.grad(chunked_loss)(coords)

    assert np.isfinite(np.asarray(grad_dense)).all()
    assert np.isfinite(np.asarray(grad_chunked)).all()
    np.testing.assert_allclose(np.asarray(grad_chunked), np.asarray(grad_dense), atol=5e-4, rtol=5e-3)

