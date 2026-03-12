from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import HDXConfig
from ..core.features import HDXFeatures
from ..core.output_index import OutputIndex
from ..core.topology import MinimalTopology
from .hdxrate import compute_kint

_AMIDE_H_NAMES = {"H", "HN", "H1", "H2", "H3", "HT1", "HT2", "HT3"}


def _find_atom_index(
    candidate_indices: np.ndarray,
    atom_names: np.ndarray,
    target_names: set[str],
) -> int:
    for idx in candidate_indices.tolist():
        if str(atom_names[idx]).upper() in target_names:
            return int(idx)
    return -1


def _is_stack_like(structure: Any) -> bool:
    return hasattr(structure, "stack_depth") and hasattr(structure, "__getitem__")


def _to_atom_array(structure: Any):
    if _is_stack_like(structure):
        return structure[0]
    if hasattr(structure, "coord") and hasattr(structure, "atom_name") and hasattr(structure, "res_name"):
        return structure
    if isinstance(structure, str):
        path = Path(structure)
        suffix = path.suffix.lower()
        if suffix == ".pdb":
            from biotite.structure.io.pdb import PDBFile, get_structure

            pdb_file = PDBFile.read(str(path))
            parsed = get_structure(pdb_file, model=1)
            return parsed[0] if _is_stack_like(parsed) else parsed
        if suffix in {".cif", ".mmcif"}:
            from biotite.structure.io.pdbx import CIFFile, get_structure

            cif_file = CIFFile.read(str(path))
            parsed = get_structure(cif_file, model=1)
            return parsed[0] if _is_stack_like(parsed) else parsed
        raise ValueError(f"Unsupported structure format: {path.suffix}")
    raise TypeError("structure must be a biotite AtomArray/AtomArrayStack or a structure file path")


def extract_coords(structure: Any) -> np.ndarray:
    if _is_stack_like(structure):
        return np.asarray(structure.coord, dtype=np.float32)
    atom_array = _to_atom_array(structure)
    return np.asarray(atom_array.coord, dtype=np.float32)


def build_exclusion_mask(
    probe_resids: np.ndarray,
    probe_chain_ids: np.ndarray,
    env_resids: np.ndarray,
    env_chain_ids: np.ndarray,
    min_sep: int = 2,
    intrachain_only: bool = False,
) -> np.ndarray:
    probe_resids = np.asarray(probe_resids, dtype=np.int32)
    env_resids = np.asarray(env_resids, dtype=np.int32)
    probe_chain_ids = np.asarray(probe_chain_ids, dtype=str)
    env_chain_ids = np.asarray(env_chain_ids, dtype=str)

    same_chain = probe_chain_ids[:, None] == env_chain_ids[None, :]
    seq_sep = np.abs(probe_resids[:, None] - env_resids[None, :])
    valid = np.where(same_chain, seq_sep > min_sep, True)
    if intrachain_only:
        valid &= same_chain

    is_env_padding = env_chain_ids[None, :] == "-1"
    is_probe_padding = probe_chain_ids[:, None] == "-1"
    valid &= ~is_env_padding
    valid &= ~is_probe_padding
    return valid.astype(np.float32)


def featurise(
    structure: Any,
    config: HDXConfig | None = None,
    output_index: OutputIndex | None = None,
) -> HDXFeatures:
    cfg = config if config is not None else HDXConfig()
    atom_array = _to_atom_array(structure)
    topology = MinimalTopology.from_biotite(atom_array)
    if output_index is None:
        output_index = OutputIndex.from_selection(topology, include_hetatm=cfg.include_hetatm)

    atom_names = topology.atom_names.astype(str)
    chain_ids = topology.chain_ids.astype(str)
    res_ids = topology.res_ids.astype(np.int32)
    atom_res_keys = topology.atom_res_keys
    residue_name_map = topology.residue_name_map()

    element_upper = np.char.upper(topology.elements.astype(str))
    heavy_mask = output_index.atom_mask & (element_upper != "H")
    heavy_atom_idx = np.flatnonzero(heavy_mask).astype(np.int32)
    backbone_o_mask = output_index.atom_mask & topology.is_backbone & (atom_names == "O")
    backbone_o_idx = np.flatnonzero(backbone_o_mask).astype(np.int32)

    prev_c_idx_per_res = np.full(topology.res_unique_ids.shape[0], -1, dtype=np.int32)
    last_c_by_chain: dict[str, int] = {}
    for i, key in enumerate(topology.res_unique_ids.tolist()):
        chain = key.split(":", 1)[0]
        prev_c_idx_per_res[i] = last_c_by_chain.get(chain, -1)
        residue_atom_idx = np.flatnonzero(atom_res_keys == key).astype(np.int32)
        c_idx = _find_atom_index(residue_atom_idx, atom_names, {"C"})
        if c_idx != -1:
            last_c_by_chain[chain] = c_idx

    amide_n_idx: list[int] = []
    amide_h_idx: list[int] = []
    amide_ca_idx: list[int] = []
    amide_c_prev_idx: list[int] = []
    amide_has_observed_h: list[bool] = []
    probe_resids: list[int] = []
    probe_chain_ids: list[str] = []
    res_keys: list[str] = []
    res_names: list[str] = []
    can_exchange: list[bool] = []

    for res_i in output_index.output_res_idx.tolist():
        if not bool(topology.res_can_exchange[res_i]):
            continue

        key = str(topology.res_unique_ids[res_i])
        residue_atom_idx = np.flatnonzero(atom_res_keys == key).astype(np.int32)
        if residue_atom_idx.size == 0:
            continue

        n_idx = _find_atom_index(residue_atom_idx, atom_names, {"N"})
        ca_idx = _find_atom_index(residue_atom_idx, atom_names, {"CA"})
        h_idx = _find_atom_index(residue_atom_idx, atom_names, _AMIDE_H_NAMES)
        prev_c_idx = int(prev_c_idx_per_res[res_i])

        if n_idx == -1 or ca_idx == -1:
            continue
        if h_idx == -1 and prev_c_idx == -1:
            continue

        chain_id, resid_text = key.split(":", 1)
        amide_n_idx.append(n_idx)
        amide_h_idx.append(n_idx if h_idx == -1 else h_idx)
        amide_ca_idx.append(ca_idx)
        amide_c_prev_idx.append(n_idx if prev_c_idx == -1 else prev_c_idx)
        amide_has_observed_h.append(h_idx != -1)
        probe_chain_ids.append(chain_id)
        probe_resids.append(int(resid_text))
        res_keys.append(key)
        res_names.append(residue_name_map[key])
        can_exchange.append(True)

    if len(amide_n_idx) == 0:
        raise ValueError("No exchangeable amide residues could be constructed from the provided structure")

    probe_resid_array = np.asarray(probe_resids, dtype=np.int32)
    probe_chain_array = np.asarray(probe_chain_ids, dtype=str)

    excl_mask_c = build_exclusion_mask(
        probe_resids=probe_resid_array,
        probe_chain_ids=probe_chain_array,
        env_resids=res_ids[heavy_atom_idx],
        env_chain_ids=chain_ids[heavy_atom_idx],
        min_sep=cfg.seq_sep_min,
        intrachain_only=cfg.intrachain_only,
    )
    excl_mask_h = build_exclusion_mask(
        probe_resids=probe_resid_array,
        probe_chain_ids=probe_chain_array,
        env_resids=res_ids[backbone_o_idx],
        env_chain_ids=chain_ids[backbone_o_idx],
        min_sep=cfg.seq_sep_min,
        intrachain_only=cfg.intrachain_only,
    )

    res_key_array = np.asarray(res_keys, dtype=str)
    res_name_array = np.asarray(res_names, dtype=str)
    can_exchange_array = np.asarray(can_exchange, dtype=bool)
    kint = None
    if cfg.use_hdxrate:
        kint = compute_kint(
            res_keys=res_key_array,
            res_names=res_name_array,
            can_exchange=can_exchange_array,
            pH=cfg.hdxrate_pH,
            temperature=cfg.hdxrate_temp,
        )

    return HDXFeatures(
        topology=topology,
        amide_N_idx=np.asarray(amide_n_idx, dtype=np.int32),
        amide_H_idx=np.asarray(amide_h_idx, dtype=np.int32),
        amide_CA_idx=np.asarray(amide_ca_idx, dtype=np.int32),
        amide_C_prev_idx=np.asarray(amide_c_prev_idx, dtype=np.int32),
        amide_has_observed_H=np.asarray(amide_has_observed_h, dtype=bool),
        heavy_atom_idx=heavy_atom_idx.astype(np.int32),
        backbone_O_idx=backbone_o_idx.astype(np.int32),
        excl_mask_c=excl_mask_c.astype(np.float32),
        excl_mask_h=excl_mask_h.astype(np.float32),
        res_keys=res_key_array,
        res_names=res_name_array,
        can_exchange=can_exchange_array,
        kint=None if kint is None else np.asarray(kint, dtype=np.float32),
    )
