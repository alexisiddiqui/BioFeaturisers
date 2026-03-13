"""Optional HDXrate integration for intrinsic rates and uptake prediction."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from biofeaturisers.config import HDXConfig
from biofeaturisers.core.topology import MinimalTopology

_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _split_res_key(res_key: str) -> tuple[str, int]:
    chain, resid = str(res_key).split(":", 1)
    return chain, int(resid)


def _residue_name_lookup(topology: MinimalTopology) -> dict[str, str]:
    names_by_key: dict[str, str] = {}
    chain_ids = np.asarray(topology.chain_ids, dtype=str)
    res_ids = np.asarray(topology.res_ids, dtype=np.int32)
    res_names = np.asarray(topology.res_names, dtype=str)
    for chain, resid, res_name in zip(chain_ids, res_ids, res_names, strict=True):
        names_by_key.setdefault(f"{chain}:{int(resid)}", str(res_name))
    return names_by_key


def _chain_index_map(res_keys: np.ndarray) -> dict[str, list[int]]:
    chain_to_indices: dict[str, list[int]] = {}
    for idx, res_key in enumerate(np.asarray(res_keys, dtype=str)):
        chain, _ = _split_res_key(res_key)
        chain_to_indices.setdefault(chain, []).append(int(idx))
    return chain_to_indices


def _to_one_letter_sequence(residue_names: Sequence[str]) -> str:
    return "".join(_THREE_TO_ONE.get(name.upper(), "X") for name in residue_names)


def _load_hdxrate_api():
    try:
        from hdxrate import k_int_from_sequence
    except ImportError as exc:  # pragma: no cover - exercised via tests with monkeypatch
        raise ImportError(
            "HDXrate integration requested but package `hdxrate` is not installed. "
            "Install `hdxrate` or set HDXConfig(use_hdxrate=False)."
        ) from exc
    return k_int_from_sequence


def compute_kint(
    topology: MinimalTopology,
    pH: float,
    temperature: float,
    config: HDXConfig | None = None,
) -> np.ndarray:
    """Compute chain-aware intrinsic rates aligned to `topology.res_unique_ids`."""
    cfg = config or HDXConfig()
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    k_int_from_sequence = _load_hdxrate_api()
    res_keys = np.asarray(topology.res_unique_ids, dtype=str)
    can_exchange = np.asarray(topology.res_can_exchange, dtype=bool)
    res_name_by_key = _residue_name_lookup(topology)
    chain_to_indices = _chain_index_map(res_keys)

    kint = np.full(res_keys.shape[0], np.nan, dtype=np.float32)
    for _, indices in chain_to_indices.items():
        chain_keys = [res_keys[i] for i in indices]
        chain_names = [res_name_by_key[k] for k in chain_keys]
        sequence = _to_one_letter_sequence(chain_names)
        rates = np.asarray(k_int_from_sequence(sequence, float(temperature), float(pH)))
        if rates.shape != (len(indices),):
            raise ValueError(
                "HDXrate returned unexpected shape "
                f"{tuple(rates.shape)} for chain sequence length {len(indices)}"
            )

        for local_idx, global_idx in enumerate(indices):
            if not bool(can_exchange[global_idx]):
                continue
            rate = float(rates[local_idx])
            if not np.isfinite(rate) or rate <= 0.0:
                continue
            kint[global_idx] = np.float32(rate)

    if not cfg.disulfide_exchange:
        # Disulfide partner detection is not encoded in MinimalTopology.
        # Any user-driven exclusion should be provided via exchange_mask.
        pass

    return kint


@jax.jit
def _predict_uptake_kernel(
    ln_pf: Array,
    kint: Array,
    can_exchange: Array,
    peptide_mask: Array,
    timepoints: Array,
) -> Array:
    k_obs = kint * jnp.exp(-ln_pf)
    d_res_t = can_exchange[None, :] * (1.0 - jnp.exp(-k_obs[None, :] * timepoints[:, None]))
    return (d_res_t @ peptide_mask.T).T


def predict_uptake(
    ln_pf: Float[Array, "n_res"],
    kint: Float[Array, "n_res"],
    can_exchange: Array,
    peptide_mask: Float[Array, "n_peptides n_res"],
    timepoints: Sequence[float] | Array,
) -> Float[Array, "n_peptides n_timepoints"]:
    """Predict per-peptide uptake curves using cached intrinsic rates."""
    ln_pf_arr = jnp.asarray(ln_pf, dtype=jnp.float32)
    kint_arr = jnp.nan_to_num(
        jnp.asarray(kint, dtype=jnp.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    can_exchange_arr = jnp.asarray(can_exchange, dtype=jnp.float32)
    peptide_mask_arr = jnp.asarray(peptide_mask, dtype=jnp.float32)
    timepoints_arr = jnp.asarray(timepoints, dtype=jnp.float32)

    if ln_pf_arr.ndim != 1:
        raise ValueError("ln_pf must be rank-1")
    if kint_arr.ndim != 1:
        raise ValueError("kint must be rank-1")
    if can_exchange_arr.ndim != 1:
        raise ValueError("can_exchange must be rank-1")
    if peptide_mask_arr.ndim != 2:
        raise ValueError("peptide_mask must be rank-2")
    if timepoints_arr.ndim != 1:
        raise ValueError("timepoints must be rank-1")

    n_res = int(ln_pf_arr.shape[0])
    if int(kint_arr.shape[0]) != n_res or int(can_exchange_arr.shape[0]) != n_res:
        raise ValueError("ln_pf, kint, and can_exchange lengths must match")
    if int(peptide_mask_arr.shape[1]) != n_res:
        raise ValueError("peptide_mask second dimension must match residue count")

    return _predict_uptake_kernel(
        ln_pf=ln_pf_arr,
        kint=kint_arr,
        can_exchange=can_exchange_arr,
        peptide_mask=peptide_mask_arr,
        timepoints=timepoints_arr,
    )


__all__ = ["compute_kint", "predict_uptake"]

