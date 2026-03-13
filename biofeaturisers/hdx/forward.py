"""HDX forward kernels for Best-Vendruscolo contact tracking."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from biofeaturisers.config import HDXConfig
from biofeaturisers.core.pairwise import dist_matrix_asymmetric
from biofeaturisers.core.switching import sigmoid_switch

from .features import HDXFeatures

_AMIDE_H_BOND_LENGTH = 1.01


def bucket_size(n_items: int) -> int:
    """Return the next power-of-2 bucket (minimum 1)."""
    if n_items <= 1:
        return 1
    return 1 << (int(n_items) - 1).bit_length()


def _pad_vector(values: Array, target_size: int, pad_value: int = 0) -> Array:
    arr = jnp.asarray(values, dtype=jnp.int32)
    pad = target_size - int(arr.shape[0])
    if pad < 0:
        raise ValueError(f"target_size {target_size} smaller than vector length {arr.shape[0]}")
    if pad == 0:
        return arr
    return jnp.pad(arr, (0, pad), mode="constant", constant_values=pad_value)


def _pad_matrix(values: Array, target_rows: int, target_cols: int) -> Array:
    arr = jnp.asarray(values, dtype=jnp.float32)
    row_pad = target_rows - int(arr.shape[0])
    col_pad = target_cols - int(arr.shape[1])
    if row_pad < 0 or col_pad < 0:
        raise ValueError(
            f"target matrix {(target_rows, target_cols)} smaller than {tuple(arr.shape)}"
        )
    if row_pad == 0 and col_pad == 0:
        return arr
    return jnp.pad(arr, ((0, row_pad), (0, col_pad)), mode="constant")


def _normalize_rows(vec: Float[Array, "n 3"]) -> Float[Array, "n 3"]:
    norm = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / jnp.maximum(norm, jnp.asarray(1e-8, dtype=vec.dtype))


def _validate_indices(name: str, idx: Array, n_atoms: int, *, allow_minus_one: bool = False) -> None:
    idx_np = np.asarray(idx, dtype=np.int64)
    if idx_np.size == 0:
        return

    min_allowed = -1 if allow_minus_one else 0
    if int(idx_np.min()) < min_allowed:
        raise ValueError(f"{name} contains out-of-range negative index")

    non_negative = idx_np[idx_np >= 0]
    if non_negative.size > 0 and int(non_negative.max()) >= int(n_atoms):
        raise ValueError(f"{name} contains index >= coords atom count ({n_atoms})")


@jax.jit
def _hdx_forward_bucketed(
    coords: Float[Array, "atom_bucket 3"],
    amide_n_idx: Array,
    amide_h_idx: Array,
    amide_ca_idx: Array,
    amide_prev_c_idx: Array,
    heavy_idx: Array,
    backbone_o_idx: Array,
    excl_mask_c: Array,
    excl_mask_h: Array,
    beta_0: float,
    beta_c: float,
    beta_h: float,
    cutoff_c: float,
    cutoff_h: float,
    steepness_c: float,
    steepness_h: float,
) -> tuple[Array, Array, Array]:
    amide_n = coords[amide_n_idx]
    amide_ca = coords[amide_ca_idx]
    amide_prev_c = coords[amide_prev_c_idx]

    v_ca = _normalize_rows(amide_ca - amide_n)
    v_prev = _normalize_rows(amide_prev_c - amide_n)
    amide_h_synth = amide_n + _AMIDE_H_BOND_LENGTH * _normalize_rows(v_ca + v_prev)

    coords_aug = jnp.concatenate((coords, amide_h_synth), axis=0)
    synth_h_idx = coords.shape[0] + jnp.arange(amide_n_idx.shape[0], dtype=jnp.int32)
    effective_h_idx = jnp.where(amide_h_idx >= 0, amide_h_idx, synth_h_idx)

    heavy = coords_aug[heavy_idx]
    backbone_o = coords_aug[backbone_o_idx]
    amide_h = coords_aug[effective_h_idx]

    dist_c = dist_matrix_asymmetric(amide_n, heavy)
    dist_h = dist_matrix_asymmetric(amide_h, backbone_o)

    nc = jnp.sum(sigmoid_switch(dist_c, cutoff_c, steepness_c) * excl_mask_c, axis=-1)
    nh = jnp.sum(sigmoid_switch(dist_h, cutoff_h, steepness_h) * excl_mask_h, axis=-1)
    ln_pf = beta_0 + beta_c * nc + beta_h * nh
    return nc, nh, ln_pf


def hdx_forward(
    coords: Float[Array, "n_atoms 3"],
    features: HDXFeatures,
    config: HDXConfig | None = None,
) -> dict[str, Float[Array, "n_probe"]]:
    """Compute BV contact counts and protection factors with power-of-2 bucketing."""
    cfg = config or HDXConfig()
    coords_arr = jnp.asarray(coords, dtype=jnp.float32)
    if coords_arr.ndim != 2 or int(coords_arr.shape[1]) != 3:
        raise ValueError(f"coords must have shape (n_atoms, 3), got {tuple(coords_arr.shape)}")

    n_atoms = int(coords_arr.shape[0])
    amide_n_idx = jnp.asarray(features.amide_N_idx, dtype=jnp.int32)
    amide_h_idx = jnp.asarray(features.amide_H_idx, dtype=jnp.int32)
    heavy_idx = jnp.asarray(features.heavy_atom_idx, dtype=jnp.int32)
    backbone_o_idx = jnp.asarray(features.backbone_O_idx, dtype=jnp.int32)
    excl_mask_c = jnp.asarray(features.excl_mask_c, dtype=jnp.float32)
    excl_mask_h = jnp.asarray(features.excl_mask_h, dtype=jnp.float32)

    n_probe = int(amide_n_idx.shape[0])
    if n_probe == 0:
        zeros = jnp.zeros((0,), dtype=coords_arr.dtype)
        return {"Nc": zeros, "Nh": zeros, "ln_Pf": zeros}

    if features.amide_CA_idx is None or features.amide_prev_C_idx is None:
        if bool(jnp.any(amide_h_idx < 0)):
            raise ValueError(
                "features include missing amide H indices but no amide_CA_idx/amide_prev_C_idx"
            )
        amide_ca_idx = amide_n_idx
        amide_prev_c_idx = amide_n_idx
    else:
        amide_ca_idx = jnp.asarray(features.amide_CA_idx, dtype=jnp.int32)
        amide_prev_c_idx = jnp.asarray(features.amide_prev_C_idx, dtype=jnp.int32)

    _validate_indices("amide_N_idx", amide_n_idx, n_atoms)
    _validate_indices("amide_H_idx", amide_h_idx, n_atoms, allow_minus_one=True)
    _validate_indices("amide_CA_idx", amide_ca_idx, n_atoms)
    _validate_indices("amide_prev_C_idx", amide_prev_c_idx, n_atoms)
    _validate_indices("heavy_atom_idx", heavy_idx, n_atoms)
    _validate_indices("backbone_O_idx", backbone_o_idx, n_atoms)

    atom_bucket = bucket_size(n_atoms)
    probe_bucket = bucket_size(n_probe)
    heavy_bucket = bucket_size(max(1, int(heavy_idx.shape[0])))
    backbone_o_bucket = bucket_size(max(1, int(backbone_o_idx.shape[0])))

    coords_pad = jnp.pad(coords_arr, ((0, atom_bucket - n_atoms), (0, 0)))
    amide_n_pad = _pad_vector(amide_n_idx, probe_bucket, pad_value=0)
    amide_h_pad = _pad_vector(amide_h_idx, probe_bucket, pad_value=-1)
    amide_ca_pad = _pad_vector(amide_ca_idx, probe_bucket, pad_value=0)
    amide_prev_c_pad = _pad_vector(amide_prev_c_idx, probe_bucket, pad_value=0)
    heavy_pad = _pad_vector(heavy_idx, heavy_bucket, pad_value=0)
    backbone_o_pad = _pad_vector(backbone_o_idx, backbone_o_bucket, pad_value=0)
    excl_mask_c_pad = _pad_matrix(excl_mask_c, probe_bucket, heavy_bucket)
    excl_mask_h_pad = _pad_matrix(excl_mask_h, probe_bucket, backbone_o_bucket)

    nc_pad, nh_pad, ln_pf_pad = _hdx_forward_bucketed(
        coords=coords_pad,
        amide_n_idx=amide_n_pad,
        amide_h_idx=amide_h_pad,
        amide_ca_idx=amide_ca_pad,
        amide_prev_c_idx=amide_prev_c_pad,
        heavy_idx=heavy_pad,
        backbone_o_idx=backbone_o_pad,
        excl_mask_c=excl_mask_c_pad,
        excl_mask_h=excl_mask_h_pad,
        beta_0=cfg.beta_0,
        beta_c=cfg.beta_c,
        beta_h=cfg.beta_h,
        cutoff_c=cfg.cutoff_c,
        cutoff_h=cfg.cutoff_h,
        steepness_c=cfg.steepness_c,
        steepness_h=cfg.steepness_h,
    )

    return {
        "Nc": nc_pad[:n_probe],
        "Nh": nh_pad[:n_probe],
        "ln_Pf": ln_pf_pad[:n_probe],
    }


__all__ = ["bucket_size", "hdx_forward"]

