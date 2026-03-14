"""Tests for experimental file and index JSON parsers."""

from __future__ import annotations

import json

import numpy as np

from biofeaturisers.io.formats import (
    load_hdx_csv,
    load_hdx_index,
    load_saxs_csv,
    load_saxs_data,
    load_saxs_index,
    parse_hdx_index_payload,
    parse_saxs_index_payload,
)


def test_load_saxs_data_from_dat_three_column(tmp_path) -> None:
    dat_path = tmp_path / "profile.dat"
    dat_path.write_text("# q I sigma\n0.10 12.0 0.4\n0.20 10.5 0.3\n", encoding="utf-8")

    parsed = load_saxs_data(dat_path)

    np.testing.assert_allclose(parsed.q, np.asarray([0.10, 0.20], dtype=np.float32))
    np.testing.assert_allclose(parsed.intensity, np.asarray([12.0, 10.5], dtype=np.float32))
    assert parsed.sigma is not None
    np.testing.assert_allclose(parsed.sigma, np.asarray([0.4, 0.3], dtype=np.float32))


def test_load_saxs_csv_with_normalized_headers(tmp_path) -> None:
    csv_path = tmp_path / "profile.csv"
    csv_path.write_text("Q Value,Intensity,Error\n0.1,9.0,0.2\n0.2,7.5,0.1\n", encoding="utf-8")

    parsed = load_saxs_csv(csv_path)

    np.testing.assert_allclose(parsed.q, np.asarray([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(parsed.intensity, np.asarray([9.0, 7.5], dtype=np.float32))
    assert parsed.sigma is not None
    np.testing.assert_allclose(parsed.sigma, np.asarray([0.2, 0.1], dtype=np.float32))


def test_load_hdx_csv_with_standard_columns(tmp_path) -> None:
    csv_path = tmp_path / "uptake.csv"
    csv_path.write_text(
        "peptide,start,end,timepoint,deuterium\nAAAG,1,4,10.0,2.3\nCCDD,5,8,60.0,4.1\n",
        encoding="utf-8",
    )

    parsed = load_hdx_csv(csv_path)

    np.testing.assert_array_equal(parsed.peptide, np.asarray(["AAAG", "CCDD"], dtype=str))
    np.testing.assert_array_equal(parsed.start, np.asarray([1, 5], dtype=np.int32))
    np.testing.assert_array_equal(parsed.end, np.asarray([4, 8], dtype=np.int32))
    np.testing.assert_allclose(parsed.timepoint, np.asarray([10.0, 60.0], dtype=np.float32))
    np.testing.assert_allclose(parsed.deuterium, np.asarray([2.3, 4.1], dtype=np.float32))


def test_parse_hdx_index_payload_supports_custom_shapes() -> None:
    list_payload = [
        {"res_key": "A:1", "res_name": "ALA", "can_exchange": False, "kint": None},
        {"res_key": "A:2", "res_name": "GLY", "can_exchange": True, "kint": 0.7},
    ]
    parsed_list = parse_hdx_index_payload(list_payload)
    np.testing.assert_array_equal(parsed_list["res_keys"], np.asarray(["A:1", "A:2"], dtype=str))
    np.testing.assert_array_equal(parsed_list["can_exchange"], np.asarray([False, True], dtype=bool))
    assert parsed_list["kint"] is not None
    np.testing.assert_allclose(parsed_list["kint"], np.asarray([np.nan, 0.7], dtype=np.float32), equal_nan=True)

    mapping_payload = {
        "A:1": {"res_name": "ALA", "can_exchange": False},
        "A:2": {"res_name": "GLY", "can_exchange": True},
    }
    parsed_map = parse_hdx_index_payload(mapping_payload)
    np.testing.assert_array_equal(parsed_map["res_names"], np.asarray(["ALA", "GLY"], dtype=str))
    assert parsed_map["kint"] is None


def test_parse_saxs_index_payload_and_file_loader(tmp_path) -> None:
    custom_payload = {
        "chains": [{"chain_id": "A", "atom_count": 12}, {"chain_id": "B", "atom_count": 4}],
        "c1": 1.02,
        "c2": 2.5,
    }
    parsed = parse_saxs_index_payload(custom_payload)
    np.testing.assert_array_equal(parsed["chain_ids"], np.asarray(["A", "B"], dtype=str))
    assert parsed["atom_counts"] == {"A": 12, "B": 4}
    assert parsed["c1_used"] == 1.02
    assert parsed["c2_used"] == 2.5

    hdx_index_path = tmp_path / "case_hdx_index.json"
    hdx_index_path.write_text(
        json.dumps(
            [
                {"res_key": "A:1", "res_name": "ALA", "can_exchange": False},
                {"res_key": "A:2", "res_name": "GLY", "can_exchange": True},
            ]
        ),
        encoding="utf-8",
    )
    loaded_hdx = load_hdx_index(hdx_index_path)
    np.testing.assert_array_equal(loaded_hdx["res_keys"], np.asarray(["A:1", "A:2"], dtype=str))

    saxs_index_path = tmp_path / "case_saxs_index.json"
    saxs_index_path.write_text(json.dumps(custom_payload), encoding="utf-8")
    loaded_saxs = load_saxs_index(saxs_index_path)
    assert loaded_saxs["atom_counts"] == {"A": 12, "B": 4}


def test_load_saxs_data_from_fit_four_columns(tmp_path) -> None:
    fit_path = tmp_path / "profile.fit"
    fit_path.write_text("0.10 9.1 0.3 9.0\n0.20 8.2 0.2 8.1\n", encoding="utf-8")

    parsed = load_saxs_data(fit_path)

    np.testing.assert_allclose(parsed.q, np.asarray([0.10, 0.20], dtype=np.float32))
    np.testing.assert_allclose(parsed.intensity, np.asarray([9.1, 8.2], dtype=np.float32))
    assert parsed.sigma is not None
    np.testing.assert_allclose(parsed.sigma, np.asarray([0.3, 0.2], dtype=np.float32))

