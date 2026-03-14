"""CLI integration tests for Typer command wiring."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from biotite.structure.io.pdb import PDBFile
from typer.testing import CliRunner

from biofeaturisers.cli import app
from biofeaturisers.io.load import load_hdx_output, load_saxs_output

runner = CliRunner()


def _fixture_structure_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "1ubq_A_1_15.pdb"


def _load_fixture_coords() -> np.ndarray:
    pdb = PDBFile.read(_fixture_structure_path())
    atom_array = pdb.get_structure(model=1)
    return np.asarray(atom_array.coord, dtype=np.float32)


def test_cli_help_shows_hdx_and_saxs_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "hdx" in result.stdout
    assert "saxs" in result.stdout


def test_hdx_cli_featurise_and_forward_roundtrip(tmp_path) -> None:
    config_path = tmp_path / "hdx.toml"
    config_path.write_text("seq_sep_min = 0\nbatch_size = 2\n", encoding="utf-8")

    feature_prefix = tmp_path / "hdx_case"
    result = runner.invoke(
        app,
        [
            "hdx",
            "featurise",
            "--structure",
            str(_fixture_structure_path()),
            "--out",
            str(feature_prefix),
            "--config",
            str(config_path),
            "--env",
            "device=cpu,disable_preallocation=true",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "hdx_case_features.npz").exists()
    assert (tmp_path / "hdx_case_topology.json").exists()

    coords_path = tmp_path / "coords.npy"
    np.save(coords_path, _load_fixture_coords())
    output_prefix = tmp_path / "hdx_run"
    result = runner.invoke(
        app,
        [
            "hdx",
            "forward",
            "--features",
            str(feature_prefix),
            "--coords",
            str(coords_path),
            "--out",
            str(output_prefix),
            "--config",
            str(config_path),
            "--env",
            "device=cpu,disable_preallocation=true",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "hdx_run_hdx_output.npz").exists()
    assert (tmp_path / "hdx_run_hdx_index.json").exists()
    arrays, index = load_hdx_output(str(output_prefix))
    assert "Nc" in arrays and "ln_Pf" in arrays
    assert arrays["Nc"].dtype == np.float32
    assert index["res_keys"].shape[0] == arrays["ln_Pf"].shape[0]
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_saxs_cli_predict_with_env_toml(tmp_path) -> None:
    saxs_config = tmp_path / "saxs.toml"
    saxs_config.write_text(
        "q_min = 0.05\nq_max = 0.30\nn_q = 18\nchunk_size = 64\nbatch_size = 2\nfit_c1_c2 = false\n",
        encoding="utf-8",
    )
    env_config = tmp_path / "env.toml"
    env_config.write_text("[compute]\ndevice = \"cpu\"\ndisable_preallocation = true\n", encoding="utf-8")

    output_prefix = tmp_path / "saxs_run"
    result = runner.invoke(
        app,
        [
            "saxs",
            "predict",
            "--structure",
            str(_fixture_structure_path()),
            "--out",
            str(output_prefix),
            "--config",
            str(saxs_config),
            "--env",
            str(env_config),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "saxs_run_saxs_output.npz").exists()
    assert (tmp_path / "saxs_run_saxs_index.json").exists()
    arrays, index = load_saxs_output(str(output_prefix))
    assert arrays["I_q"].shape[0] == 18
    assert index["c1_used"] is not None
    assert index["c2_used"] is not None
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

