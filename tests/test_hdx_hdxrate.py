import importlib.util

import numpy as np
import pytest

from biofeaturisers.hdx.hdxrate import compute_kint, predict_uptake


def test_compute_kint_requires_optional_hdxrate_dependency():
    if importlib.util.find_spec("hdxrate") is not None:
        pytest.skip("hdxrate is installed in this environment")

    with pytest.raises(ImportError):
        compute_kint(
            res_keys=np.array(["A:1", "A:2"], dtype=str),
            res_names=np.array(["ALA", "GLY"], dtype=str),
            can_exchange=np.array([False, True], dtype=bool),
            pH=7.0,
            temperature=298.15,
        )


def test_predict_uptake_matches_closed_form_examples():
    ln_pf = np.array([10.0, 0.0], dtype=np.float32)
    kint = np.array([1.0, 1.0], dtype=np.float32)
    can_exchange = np.array([1.0, 1.0], dtype=np.float32)
    peptide_mask = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    uptake = np.asarray(
        predict_uptake(
            ln_Pf=ln_pf,
            kint=kint,
            can_exchange=can_exchange,
            peptide_mask=peptide_mask,
            timepoints=(1.0,),
        )
    )[:, 0]

    expected_protected = 1.0 - np.exp(-np.exp(-10.0))
    expected_exposed = 1.0 - np.exp(-1.0)
    expected = np.array(
        [
            expected_protected,
            expected_exposed,
            expected_protected + expected_exposed,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(uptake, expected, rtol=1e-6, atol=1e-6)
