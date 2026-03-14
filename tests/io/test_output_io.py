"""Tests for HDX/SAXS output persistence helpers."""

from __future__ import annotations

import numpy as np

from biofeaturisers.io.load import load_hdx_output, load_saxs_output
from biofeaturisers.io.save import save_hdx_output, save_saxs_output


def test_hdx_output_roundtrip_with_index_metadata(tmp_path) -> None:
    prefix = tmp_path / "hdx_case"
    save_hdx_output(
        str(prefix),
        nc=np.asarray([1.0, 2.0], dtype=np.float64),
        nh=np.asarray([0.5, 1.5], dtype=np.float64),
        ln_pf=np.asarray([3.0, 4.0], dtype=np.float64),
        uptake_curves=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        res_keys=np.asarray(["A:1", "A:2"], dtype=str),
        res_names=np.asarray(["ALA", "GLY"], dtype=str),
        can_exchange=np.asarray([False, True], dtype=bool),
        kint=np.asarray([np.nan, 1.2], dtype=np.float64),
    )

    arrays, index = load_hdx_output(str(prefix))

    np.testing.assert_allclose(arrays["Nc"], np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(arrays["Nh"], np.asarray([0.5, 1.5], dtype=np.float32))
    np.testing.assert_allclose(arrays["ln_Pf"], np.asarray([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(
        arrays["uptake_curves"], np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    )
    assert arrays["Nc"].dtype == np.float32
    assert arrays["ln_Pf"].dtype == np.float32
    np.testing.assert_array_equal(index["res_keys"], np.asarray(["A:1", "A:2"], dtype=str))
    np.testing.assert_array_equal(index["res_names"], np.asarray(["ALA", "GLY"], dtype=str))
    np.testing.assert_array_equal(index["can_exchange"], np.asarray([False, True], dtype=bool))
    assert index["kint"] is not None
    np.testing.assert_allclose(index["kint"], np.asarray([np.nan, 1.2], dtype=np.float32), equal_nan=True)


def test_saxs_output_roundtrip_builds_chain_counts(tmp_path) -> None:
    prefix = tmp_path / "saxs_case"
    save_saxs_output(
        str(prefix),
        i_q=np.asarray([10.0, 8.0, 6.0], dtype=np.float64),
        q_values=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        partials=np.ones((6, 3), dtype=np.float64),
        chain_ids=np.asarray(["A", "A", "B", "A"], dtype=str),
        c1_used=1.05,
        c2_used=2.2,
    )

    arrays, index = load_saxs_output(str(prefix))

    np.testing.assert_allclose(arrays["I_q"], np.asarray([10.0, 8.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(arrays["q_values"], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    assert arrays["partials"].dtype == np.float32
    np.testing.assert_array_equal(index["chain_ids"], np.asarray(["A", "B"], dtype=str))
    assert index["atom_counts"] == {"A": 3, "B": 1}
    assert index["c1_used"] == 1.05
    assert index["c2_used"] == 2.2

