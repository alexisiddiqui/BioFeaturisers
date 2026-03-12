from __future__ import annotations

from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
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


def _to_one_letter(res_names: np.ndarray) -> str:
    letters: list[str] = []
    for res_name in res_names:
        key = str(res_name).upper()
        if key not in _THREE_TO_ONE:
            raise ValueError(f"Unsupported residue for HDXrate sequence conversion: {key}")
        letters.append(_THREE_TO_ONE[key])
    return "".join(letters)


def compute_kint(
    res_keys: np.ndarray,
    res_names: np.ndarray,
    can_exchange: np.ndarray,
    pH: float,
    temperature: float,
) -> np.ndarray:
    """
    Compute intrinsic exchange rates with one HDXrate call per chain.

    Returns a residue-aligned array with np.nan for non-exchangeable positions.
    """
    try:
        from hdxrate import k_int_from_sequence
    except ImportError as exc:
        raise ImportError(
            "HDXConfig.use_hdxrate=True requires the optional 'hdxrate' package to be installed."
        ) from exc

    if res_keys.shape != res_names.shape or res_keys.shape != can_exchange.shape:
        raise ValueError("res_keys, res_names and can_exchange must have identical shape")

    chain_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(res_keys.tolist()):
        chain_to_indices[str(key).split(":", 1)[0]].append(idx)

    kint = np.full(res_keys.shape[0], np.nan, dtype=np.float32)
    for chain_id in sorted(chain_to_indices):
        idxs = np.asarray(chain_to_indices[chain_id], dtype=np.int32)
        chain_names = res_names[idxs]
        sequence = _to_one_letter(chain_names)
        rates = np.asarray(k_int_from_sequence(sequence, temperature, pH), dtype=np.float32)
        if rates.shape[0] != idxs.shape[0]:
            raise ValueError(
                f"HDXrate returned {rates.shape[0]} rates for chain {chain_id}, expected {idxs.shape[0]}"
            )

        for local_i, global_i in enumerate(idxs.tolist()):
            if not bool(can_exchange[global_i]):
                kint[global_i] = np.nan
                continue
            rate = float(rates[local_i])
            kint[global_i] = np.nan if rate <= 0.0 else np.float32(rate)
    return kint


@partial(jax.jit, static_argnames=("timepoints",))
def predict_uptake(
    ln_Pf: Float[Array, "N_res"],
    kint: Float[Array, "N_res"],
    can_exchange: Float[Array, "N_res"],
    peptide_mask: Float[Array, "N_peptides N_res"],
    timepoints: tuple[float, ...],
) -> Float[Array, "N_peptides N_timepoints"]:
    """
    D(t) = Σ_k can_exchange_k * (1 - exp(-kint_k * exp(-ln_Pf_k) * t))
    """
    kint_safe = jnp.nan_to_num(kint, nan=0.0, posinf=0.0, neginf=0.0)
    can_ex = jnp.asarray(can_exchange, dtype=ln_Pf.dtype)
    pep_mask = jnp.asarray(peptide_mask, dtype=ln_Pf.dtype)

    def uptake_at_time(t: float) -> Float[Array, "N_peptides"]:
        k_eff = kint_safe * jnp.exp(-ln_Pf)
        d_res = can_ex * (1.0 - jnp.exp(-k_eff * t))
        return pep_mask @ d_res

    return jnp.stack([uptake_at_time(t) for t in timepoints], axis=-1)
