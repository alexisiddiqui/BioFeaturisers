"""Persistence helpers for feature/topology bundles and index arrays."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology


def _feature_paths(prefix: str) -> tuple[Path, Path]:
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    features_path = prefix_path.parent / f"{prefix_path.name}_features.npz"
    topology_path = prefix_path.parent / f"{prefix_path.name}_topology.json"
    return features_path, topology_path


def _output_paths(prefix: str, kind: str) -> tuple[Path, Path]:
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = prefix_path.parent / f"{prefix_path.name}_{kind}_output.npz"
    index_path = prefix_path.parent / f"{prefix_path.name}_{kind}_index.json"
    return output_path, index_path


def output_index_arrays(output_index: OutputIndex) -> dict[str, np.ndarray]:
    """Convert ``OutputIndex`` fields to deterministic numpy arrays."""
    return {
        "output_atom_mask": np.asarray(output_index.atom_mask, dtype=bool),
        "output_probe_mask": np.asarray(output_index.probe_mask, dtype=bool),
        "output_mask": np.asarray(output_index.output_mask, dtype=bool),
        "output_atom_idx": np.asarray(output_index.atom_idx, dtype=np.int32),
        "output_probe_idx": np.asarray(output_index.probe_idx, dtype=np.int32),
        "output_res_idx": np.asarray(output_index.output_res_idx, dtype=np.int32),
    }


def save_feature_bundle(
    prefix: str, *, topology: MinimalTopology, arrays: Mapping[str, np.ndarray]
) -> None:
    """Persist static feature arrays and topology as ``.npz + .json``."""
    features_path, topology_path = _feature_paths(prefix)
    np.savez(features_path, **{key: np.asarray(value) for key, value in arrays.items()})
    with topology_path.open("w", encoding="utf-8") as handle:
        json.dump(topology.to_json(), handle)


def _chain_counts(chain_ids: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chain_id in np.asarray(chain_ids, dtype=str):
        counts[str(chain_id)] = counts.get(str(chain_id), 0) + 1
    return counts


def save_hdx_output(
    prefix: str,
    *,
    nc: np.ndarray,
    nh: np.ndarray,
    ln_pf: np.ndarray,
    res_keys: np.ndarray,
    res_names: np.ndarray,
    can_exchange: np.ndarray,
    kint: np.ndarray | None = None,
    uptake_curves: np.ndarray | None = None,
) -> None:
    """Persist HDX forward outputs and output-index metadata."""
    output_path, index_path = _output_paths(prefix, "hdx")
    nc_arr = np.asarray(nc, dtype=np.float32)
    nh_arr = np.asarray(nh, dtype=np.float32)
    ln_pf_arr = np.asarray(ln_pf, dtype=np.float32)
    if nc_arr.ndim != 1 or nh_arr.ndim != 1 or ln_pf_arr.ndim != 1:
        raise ValueError("HDX outputs Nc/Nh/ln_Pf must all be rank-1")
    if nc_arr.shape != nh_arr.shape or nc_arr.shape != ln_pf_arr.shape:
        raise ValueError("HDX outputs Nc/Nh/ln_Pf must have identical shapes")

    res_keys_arr = np.asarray(res_keys, dtype=str)
    res_names_arr = np.asarray(res_names, dtype=str)
    can_exchange_arr = np.asarray(can_exchange, dtype=bool)
    if res_keys_arr.shape != ln_pf_arr.shape:
        raise ValueError("res_keys length must match HDX output length")
    if res_names_arr.shape != ln_pf_arr.shape:
        raise ValueError("res_names length must match HDX output length")
    if can_exchange_arr.shape != ln_pf_arr.shape:
        raise ValueError("can_exchange length must match HDX output length")

    kint_arr = None if kint is None else np.asarray(kint, dtype=np.float32)
    if kint_arr is not None and kint_arr.shape != ln_pf_arr.shape:
        raise ValueError("kint length must match HDX output length when provided")

    arrays = {
        "Nc": nc_arr,
        "Nh": nh_arr,
        "ln_Pf": ln_pf_arr,
    }
    if uptake_curves is not None:
        arrays["uptake_curves"] = np.asarray(uptake_curves, dtype=np.float32)
    np.savez(output_path, **arrays)

    index_payload: dict[str, object] = {
        "res_keys": res_keys_arr.tolist(),
        "res_names": res_names_arr.tolist(),
        "can_exchange": can_exchange_arr.tolist(),
        "kint": None,
    }
    if kint_arr is not None:
        index_payload["kint"] = kint_arr.tolist()
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle)


def save_saxs_output(
    prefix: str,
    *,
    i_q: np.ndarray,
    q_values: np.ndarray,
    chain_ids: np.ndarray,
    c1_used: float | None = None,
    c2_used: float | None = None,
    partials: np.ndarray | None = None,
    atom_counts: Mapping[str, int] | None = None,
) -> None:
    """Persist SAXS outputs and output-index metadata."""
    output_path, index_path = _output_paths(prefix, "saxs")
    i_q_arr = np.asarray(i_q, dtype=np.float32)
    q_values_arr = np.asarray(q_values, dtype=np.float32)
    if i_q_arr.ndim != 1 or q_values_arr.ndim != 1:
        raise ValueError("SAXS output I_q and q_values must be rank-1")
    if i_q_arr.shape != q_values_arr.shape:
        raise ValueError("I_q and q_values must have matching shapes")

    arrays = {
        "I_q": i_q_arr,
        "q_values": q_values_arr,
    }
    if partials is not None:
        partials_arr = np.asarray(partials, dtype=np.float32)
        if partials_arr.ndim != 2 or int(partials_arr.shape[1]) != int(i_q_arr.shape[0]):
            raise ValueError("partials must have shape (n_terms, n_q)")
        arrays["partials"] = partials_arr
    np.savez(output_path, **arrays)

    chain_ids_arr = np.asarray(chain_ids, dtype=str)
    unique_chain_ids = list(dict.fromkeys(chain_ids_arr.tolist()))
    chain_counts = (
        {str(key): int(value) for key, value in atom_counts.items()}
        if atom_counts is not None
        else _chain_counts(chain_ids_arr)
    )
    index_payload = {
        "chain_ids": unique_chain_ids,
        "atom_counts": chain_counts,
        "c1_used": None if c1_used is None else float(c1_used),
        "c2_used": None if c2_used is None else float(c2_used),
    }
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle)
