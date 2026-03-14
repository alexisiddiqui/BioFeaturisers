"""Optional JIT/bucketing behavior tests."""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from biotite.structure.io.pdb import PDBFile

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx import forward as hdx_forward_module
from biofeaturisers.hdx.featurise import featurise
from biofeaturisers.hdx.forward import bucket_size, hdx_forward

_RUN_SLOW = os.getenv("BIOFEATURISERS_RUN_SLOW", "0") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN_SLOW,
    reason="Set BIOFEATURISERS_RUN_SLOW=1 to run stress/JIT optional suites.",
)


def _load_1ubq_fragment():
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "1ubq_A_1_15.pdb"
    pdb = PDBFile.read(fixture_path)
    return pdb.get_structure(model=1)


def test_bucket_size_power_of_two_contract() -> None:
    for n_items in [100, 500, 1000, 2000, 5000]:
        bucket = bucket_size(n_items)
        assert bucket >= n_items
        assert bucket & (bucket - 1) == 0


def test_hdx_forward_reuses_compile_for_same_bucket() -> None:
    atom_array = _load_1ubq_fragment()
    config = HDXConfig(seq_sep_min=0)
    features = featurise(atom_array, config=config)
    coords = np.asarray(atom_array.coord, dtype=np.float32)

    assert bucket_size(coords.shape[0]) == bucket_size(coords.shape[0] + 3)
    coords_plus3 = np.concatenate(
        [coords, np.asarray([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float32)],
        axis=0,
    )

    _ = hdx_forward(jnp.asarray(coords, dtype=jnp.float32), features=features, config=config)
    cache_after_first = hdx_forward_module._hdx_forward_dense._cache_size()

    _ = hdx_forward(jnp.asarray(coords_plus3, dtype=jnp.float32), features=features, config=config)
    cache_after_second = hdx_forward_module._hdx_forward_dense._cache_size()
    assert cache_after_second == cache_after_first


def test_hdx_forward_static_shape_consistency_with_padding() -> None:
    atom_array = _load_1ubq_fragment()
    config = HDXConfig(seq_sep_min=0)
    features = featurise(atom_array, config=config)
    coords = np.asarray(atom_array.coord, dtype=np.float32)

    target = bucket_size(coords.shape[0])
    n_pad = target - coords.shape[0]
    coords_pad = np.concatenate([coords, np.zeros((n_pad, 3), dtype=np.float32)], axis=0)

    base = hdx_forward(jnp.asarray(coords, dtype=jnp.float32), features=features, config=config)
    padded = hdx_forward(jnp.asarray(coords_pad, dtype=jnp.float32), features=features, config=config)
    np.testing.assert_allclose(np.asarray(padded["Nc"]), np.asarray(base["Nc"]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(padded["Nh"]), np.asarray(base["Nh"]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(padded["ln_Pf"]), np.asarray(base["ln_Pf"]), atol=1e-6)

