"""HDX featurisation from Biotite topology objects."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from biotite.structure import AtomArray

from biofeaturisers.config import HDXConfig
from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology

from .features import HDXFeatures

_AMIDE_H_NAMES = ("H", "HN", "H1", "H2", "H3", "HT1", "HT2", "HT3", "1H", "2H", "3H")


def _split_res_key(res_key: str) -> tuple[str, int]:
    chain, resid = str(res_key).split(":", 1)
    return chain, int(resid)


def _residue_name_lookup(topology: MinimalTopology) -> dict[str, str]:
    lookup: dict[str, str] = {}
    chain_ids = np.asarray(topology.chain_ids, dtype=str)
    res_ids = np.asarray(topology.res_ids, dtype=np.int32)
    res_names = np.asarray(topology.res_names, dtype=str)
    for chain, resid, res_name in zip(chain_ids, res_ids, res_names, strict=True):
        key = f"{chain}:{int(resid)}"
        lookup.setdefault(key, str(res_name))
    return lookup


def _previous_residue_lookup(res_keys: np.ndarray) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    per_chain_prev: dict[str, str] = {}
    for key in np.asarray(res_keys, dtype=str):
        chain, _ = _split_res_key(key)
        previous[key] = per_chain_prev.get(chain)
        per_chain_prev[chain] = key
    return previous


def _find_atom_index(
    *,
    atom_mask: np.ndarray,
    atom_names: np.ndarray,
    chain_ids: np.ndarray,
    res_ids: np.ndarray,
    chain: str,
    resid: int,
    names: Iterable[str],
) -> int:
    residue_mask = atom_mask & (chain_ids == chain) & (res_ids == resid)
    if not np.any(residue_mask):
        return -1
    for atom_name in names:
        idx = np.flatnonzero(residue_mask & (atom_names == atom_name))
        if idx.size > 0:
            return int(idx[0])
    return -1


def build_exclusion_mask(
    probe_resids: np.ndarray,
    probe_chain_ids: np.ndarray,
    env_resids: np.ndarray,
    env_chain_ids: np.ndarray,
    min_sep: int = 2,
    intrachain_only: bool = False,
) -> np.ndarray:
    """Build sequence/chain exclusion mask for probe-environment contacts."""
    if min_sep < 0:
        raise ValueError("min_sep must be >= 0")

    probe_res = np.asarray(probe_resids, dtype=np.int32)
    probe_chain = np.asarray(probe_chain_ids, dtype=str)
    env_res = np.asarray(env_resids, dtype=np.int32)
    env_chain = np.asarray(env_chain_ids, dtype=str)

    if probe_res.ndim != 1 or probe_chain.ndim != 1:
        raise ValueError("probe_resids and probe_chain_ids must be rank-1")
    if env_res.ndim != 1 or env_chain.ndim != 1:
        raise ValueError("env_resids and env_chain_ids must be rank-1")
    if probe_res.shape[0] != probe_chain.shape[0]:
        raise ValueError("probe_resids and probe_chain_ids must have equal length")
    if env_res.shape[0] != env_chain.shape[0]:
        raise ValueError("env_resids and env_chain_ids must have equal length")

    same_chain = probe_chain[:, None] == env_chain[None, :]
    seq_sep = np.abs(probe_res[:, None] - env_res[None, :])
    too_close = same_chain & (seq_sep <= min_sep)

    valid = ~too_close
    if intrachain_only:
        valid &= same_chain

    probe_padding = probe_chain[:, None] == "-1"
    env_padding = env_chain[None, :] == "-1"
    valid &= ~probe_padding
    valid &= ~env_padding
    return valid.astype(np.float32)


def featurise(
    atom_array: AtomArray,
    config: HDXConfig | None = None,
    output_index: OutputIndex | None = None,
    include_chains: list[str] | None = None,
    exclude_chains: list[str] | None = None,
) -> HDXFeatures:
    """Create `HDXFeatures` from a Biotite `AtomArray`."""
    if not isinstance(atom_array, AtomArray):
        raise TypeError("featurise expects a biotite.structure.AtomArray input")

    cfg = config or HDXConfig()
    topology = MinimalTopology.from_biotite(atom_array, exchange_mask=cfg.exchange_mask)
    if output_index is None:
        output_index = OutputIndex.from_selection(
            topology,
            include_chains=include_chains,
            exclude_chains=exclude_chains,
            include_hetatm=cfg.include_hetatm,
        )

    atom_names = np.asarray(topology.atom_names, dtype=str)
    chain_ids = np.asarray(topology.chain_ids, dtype=str)
    res_ids = np.asarray(topology.res_ids, dtype=np.int32)
    element = np.char.upper(np.asarray(topology.element, dtype=str))
    is_hetatm = np.asarray(topology.is_hetatm, dtype=bool)
    env_atom_mask = np.asarray(output_index.atom_mask, dtype=bool)
    probe_atom_mask = np.asarray(output_index.probe_mask, dtype=bool)

    res_unique_ids = np.asarray(topology.res_unique_ids, dtype=str)
    output_res_idx = np.asarray(output_index.output_res_idx, dtype=np.int32)
    res_keys = res_unique_ids[output_res_idx]
    res_name_by_key = _residue_name_lookup(topology)
    res_names = np.asarray([res_name_by_key.get(key, "UNK") for key in res_keys], dtype=str)
    can_exchange = np.asarray(topology.res_can_exchange, dtype=bool)[output_res_idx]

    prev_key_by_key = _previous_residue_lookup(res_unique_ids)

    amide_n_idx: list[int] = []
    amide_h_idx: list[int] = []
    amide_ca_idx: list[int] = []
    amide_prev_c_idx: list[int] = []
    probe_resids: list[int] = []
    probe_chain_ids: list[str] = []

    for res_key, is_exchangeable in zip(res_keys, can_exchange, strict=True):
        if not bool(is_exchangeable):
            continue

        chain, resid = _split_res_key(res_key)
        n_idx = _find_atom_index(
            atom_mask=probe_atom_mask,
            atom_names=atom_names,
            chain_ids=chain_ids,
            res_ids=res_ids,
            chain=chain,
            resid=resid,
            names=("N",),
        )
        if n_idx < 0:
            raise ValueError(f"Missing backbone N atom for exchangeable residue {res_key}")

        ca_idx = _find_atom_index(
            atom_mask=probe_atom_mask,
            atom_names=atom_names,
            chain_ids=chain_ids,
            res_ids=res_ids,
            chain=chain,
            resid=resid,
            names=("CA",),
        )
        if ca_idx < 0:
            raise ValueError(f"Missing backbone CA atom for exchangeable residue {res_key}")

        h_idx = _find_atom_index(
            atom_mask=probe_atom_mask,
            atom_names=atom_names,
            chain_ids=chain_ids,
            res_ids=res_ids,
            chain=chain,
            resid=resid,
            names=_AMIDE_H_NAMES,
        )

        prev_key = prev_key_by_key.get(str(res_key))
        prev_c_idx = -1
        if prev_key is not None:
            prev_chain, prev_resid = _split_res_key(prev_key)
            prev_c_idx = _find_atom_index(
                atom_mask=probe_atom_mask,
                atom_names=atom_names,
                chain_ids=chain_ids,
                res_ids=res_ids,
                chain=prev_chain,
                resid=prev_resid,
                names=("C",),
            )
        if h_idx < 0 and prev_c_idx < 0:
            raise ValueError(
                f"Cannot analytically place amide H for residue {res_key}: "
                "missing previous-residue backbone C atom"
            )
        if prev_c_idx < 0:
            prev_c_idx = n_idx

        amide_n_idx.append(n_idx)
        amide_h_idx.append(h_idx)
        amide_ca_idx.append(ca_idx)
        amide_prev_c_idx.append(prev_c_idx)
        probe_resids.append(resid)
        probe_chain_ids.append(chain)

    heavy_atom_mask = env_atom_mask & (element != "H")
    heavy_atom_idx = np.flatnonzero(heavy_atom_mask).astype(np.int32)
    heavy_resids = res_ids[heavy_atom_idx]
    heavy_chain_ids = chain_ids[heavy_atom_idx]

    backbone_o_mask = env_atom_mask & (~is_hetatm) & np.isin(atom_names, np.asarray(["O", "OXT"]))
    backbone_o_idx = np.flatnonzero(backbone_o_mask).astype(np.int32)
    backbone_o_resids = res_ids[backbone_o_idx]
    backbone_o_chain_ids = chain_ids[backbone_o_idx]

    if len(probe_resids) == 0:
        excl_mask_c = np.zeros((0, heavy_atom_idx.shape[0]), dtype=np.float32)
        excl_mask_h = np.zeros((0, backbone_o_idx.shape[0]), dtype=np.float32)
    else:
        excl_mask_c = build_exclusion_mask(
            probe_resids=np.asarray(probe_resids, dtype=np.int32),
            probe_chain_ids=np.asarray(probe_chain_ids, dtype=str),
            env_resids=heavy_resids,
            env_chain_ids=heavy_chain_ids,
            min_sep=cfg.seq_sep_min,
            intrachain_only=cfg.intrachain_only,
        )
        excl_mask_h = build_exclusion_mask(
            probe_resids=np.asarray(probe_resids, dtype=np.int32),
            probe_chain_ids=np.asarray(probe_chain_ids, dtype=str),
            env_resids=backbone_o_resids,
            env_chain_ids=backbone_o_chain_ids,
            min_sep=cfg.seq_sep_min,
            intrachain_only=cfg.intrachain_only,
        )

    kint = None
    if cfg.use_hdxrate:
        from .hdxrate import compute_kint

        kint_all = compute_kint(
            topology=topology,
            pH=cfg.hdxrate_pH,
            temperature=cfg.hdxrate_temp,
            config=cfg,
        )
        kint = kint_all[output_res_idx]

    return HDXFeatures(
        topology=topology,
        output_index=output_index,
        amide_N_idx=np.asarray(amide_n_idx, dtype=np.int32),
        amide_H_idx=np.asarray(amide_h_idx, dtype=np.int32),
        amide_CA_idx=np.asarray(amide_ca_idx, dtype=np.int32),
        amide_prev_C_idx=np.asarray(amide_prev_c_idx, dtype=np.int32),
        heavy_atom_idx=heavy_atom_idx,
        backbone_O_idx=backbone_o_idx,
        excl_mask_c=excl_mask_c,
        excl_mask_h=excl_mask_h,
        res_keys=res_keys,
        res_names=res_names,
        can_exchange=can_exchange,
        kint=kint,
    )


__all__ = ["build_exclusion_mask", "featurise"]

