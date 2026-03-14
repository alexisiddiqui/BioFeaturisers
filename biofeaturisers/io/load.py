"""Loading helpers for feature/topology bundles and output index state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology
from biofeaturisers.io.formats import load_hdx_index, load_saxs_index


def _feature_paths(prefix: str) -> tuple[Path, Path]:
    prefix_path = Path(prefix)
    features_path = prefix_path.parent / f"{prefix_path.name}_features.npz"
    topology_path = prefix_path.parent / f"{prefix_path.name}_topology.json"
    return features_path, topology_path


def _output_paths(prefix: str, kind: str) -> tuple[Path, Path]:
    prefix_path = Path(prefix)
    output_path = prefix_path.parent / f"{prefix_path.name}_{kind}_output.npz"
    index_path = prefix_path.parent / f"{prefix_path.name}_{kind}_index.json"
    return output_path, index_path


def load_feature_bundle(prefix: str) -> tuple[MinimalTopology, dict[str, np.ndarray]]:
    """Load static feature arrays and topology from ``.npz + .json``."""
    features_path, topology_path = _feature_paths(prefix)
    with topology_path.open("r", encoding="utf-8") as handle:
        topology = MinimalTopology.from_json(json.load(handle))

    with np.load(features_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return topology, arrays


def output_index_from_arrays(arrays: Mapping[str, np.ndarray]) -> OutputIndex:
    """Reconstruct ``OutputIndex`` from serialized array payload."""
    return OutputIndex(
        atom_mask=arrays["output_atom_mask"],
        probe_mask=arrays["output_probe_mask"],
        output_mask=arrays["output_mask"],
        atom_idx=arrays["output_atom_idx"],
        probe_idx=arrays["output_probe_idx"],
        output_res_idx=arrays["output_res_idx"],
    )


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_hdx_output(prefix: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load HDX forward outputs and index metadata."""
    output_path, index_path = _output_paths(prefix, "hdx")
    arrays = _load_npz_arrays(output_path)
    index = load_hdx_index(index_path)
    return arrays, index


def load_saxs_output(prefix: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load SAXS forward outputs and index metadata."""
    output_path, index_path = _output_paths(prefix, "saxs")
    arrays = _load_npz_arrays(output_path)
    index = load_saxs_index(index_path)
    return arrays, index
