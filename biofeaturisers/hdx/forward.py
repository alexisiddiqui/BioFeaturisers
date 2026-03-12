from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..config import HDXConfig
from ..core.features import HDXFeatures
from ..core.pairwise import dist_matrix_asymmetric
from ..core.switching import sigmoid_switch


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pad_1d(values: np.ndarray, bucket: int, pad_value: int | bool = 0) -> jax.Array:
    padded = np.full((bucket,), pad_value, dtype=values.dtype)
    padded[: values.shape[0]] = values
    return jnp.asarray(padded)


def _pad_2d(values: np.ndarray, rows: int, cols: int, pad_value: float = 0.0) -> jax.Array:
    padded = np.full((rows, cols), pad_value, dtype=values.dtype)
    padded[: values.shape[0], : values.shape[1]] = values
    return jnp.asarray(padded)


def _unit(v: jax.Array) -> jax.Array:
    return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _analytic_amide_h(n_xyz: jax.Array, ca_xyz: jax.Array, c_prev_xyz: jax.Array) -> jax.Array:
    v1 = _unit(ca_xyz - n_xyz)
    v2 = _unit(c_prev_xyz - n_xyz)
    bisector = _unit(v1 + v2)
    return n_xyz + 1.01 * bisector


@partial(
    jax.jit,
    static_argnames=("beta_c", "beta_h", "beta_0", "cutoff_c", "cutoff_h", "steepness"),
)
def _hdx_forward_kernel(
    coords: jax.Array,
    amide_n_idx: jax.Array,
    amide_h_idx: jax.Array,
    amide_ca_idx: jax.Array,
    amide_c_prev_idx: jax.Array,
    amide_has_observed_h: jax.Array,
    heavy_idx: jax.Array,
    backbone_o_idx: jax.Array,
    excl_mask_c: jax.Array,
    excl_mask_h: jax.Array,
    beta_c: float,
    beta_h: float,
    beta_0: float,
    cutoff_c: float,
    cutoff_h: float,
    steepness: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_xyz = coords[amide_n_idx]
    observed_h_xyz = coords[amide_h_idx]
    virtual_h_xyz = _analytic_amide_h(n_xyz, coords[amide_ca_idx], coords[amide_c_prev_idx])
    h_xyz = jnp.where(amide_has_observed_h[:, None], observed_h_xyz, virtual_h_xyz)

    heavy_xyz = coords[heavy_idx]
    bb_o_xyz = coords[backbone_o_idx]

    dist_c = dist_matrix_asymmetric(n_xyz, heavy_xyz)
    dist_h = dist_matrix_asymmetric(h_xyz, bb_o_xyz)

    mask_c = jnp.asarray(excl_mask_c, dtype=coords.dtype)
    mask_h = jnp.asarray(excl_mask_h, dtype=coords.dtype)
    nc = jnp.sum(sigmoid_switch(dist_c, r0=cutoff_c, b=steepness) * mask_c, axis=-1)
    nh = jnp.sum(sigmoid_switch(dist_h, r0=cutoff_h, b=steepness) * mask_h, axis=-1)
    ln_pf = beta_0 + beta_c * nc + beta_h * nh
    return nc, nh, ln_pf


def hdx_forward(coords: np.ndarray | jax.Array, features: HDXFeatures, config: HDXConfig) -> dict[str, jax.Array]:
    coords_array = jnp.asarray(coords, dtype=jnp.float32)
    if coords_array.ndim != 2 or coords_array.shape[1] != 3:
        raise ValueError("coords must have shape (N_atoms, 3)")

    n_atoms = int(coords_array.shape[0])
    n_probe = int(features.amide_N_idx.shape[0])
    n_heavy = int(features.heavy_atom_idx.shape[0])
    n_bb_o = int(features.backbone_O_idx.shape[0])

    atom_bucket = _next_power_of_two(max(n_atoms, 1))
    probe_bucket = _next_power_of_two(max(n_probe, 1))
    heavy_bucket = _next_power_of_two(max(n_heavy, 1))
    o_bucket = _next_power_of_two(max(n_bb_o, 1))

    pad_atoms = atom_bucket - n_atoms
    coords_padded = jnp.pad(coords_array, ((0, pad_atoms), (0, 0)))

    amide_n_idx = _pad_1d(features.amide_N_idx.astype(np.int32), probe_bucket, pad_value=0)
    amide_h_idx = _pad_1d(features.amide_H_idx.astype(np.int32), probe_bucket, pad_value=0)
    amide_ca_idx = _pad_1d(features.amide_CA_idx.astype(np.int32), probe_bucket, pad_value=0)
    amide_c_prev_idx = _pad_1d(features.amide_C_prev_idx.astype(np.int32), probe_bucket, pad_value=0)
    amide_has_observed_h = _pad_1d(
        features.amide_has_observed_H.astype(bool),
        probe_bucket,
        pad_value=False,
    )
    heavy_idx = _pad_1d(features.heavy_atom_idx.astype(np.int32), heavy_bucket, pad_value=0)
    backbone_o_idx = _pad_1d(features.backbone_O_idx.astype(np.int32), o_bucket, pad_value=0)
    excl_mask_c = _pad_2d(features.excl_mask_c.astype(np.float32), probe_bucket, heavy_bucket, pad_value=0.0)
    excl_mask_h = _pad_2d(features.excl_mask_h.astype(np.float32), probe_bucket, o_bucket, pad_value=0.0)

    nc, nh, ln_pf = _hdx_forward_kernel(
        coords=coords_padded,
        amide_n_idx=amide_n_idx,
        amide_h_idx=amide_h_idx,
        amide_ca_idx=amide_ca_idx,
        amide_c_prev_idx=amide_c_prev_idx,
        amide_has_observed_h=amide_has_observed_h,
        heavy_idx=heavy_idx,
        backbone_o_idx=backbone_o_idx,
        excl_mask_c=excl_mask_c,
        excl_mask_h=excl_mask_h,
        beta_c=config.beta_c,
        beta_h=config.beta_h,
        beta_0=config.beta_0,
        cutoff_c=config.cutoff_c,
        cutoff_h=config.cutoff_h,
        steepness=config.steepness,
    )
    return {"Nc": nc[:n_probe], "Nh": nh[:n_probe], "ln_Pf": ln_pf[:n_probe]}


def forward(
    coords: np.ndarray | jax.Array,
    features: HDXFeatures,
    config: HDXConfig,
    weights: np.ndarray | jax.Array | None = None,
) -> dict[str, jax.Array]:
    coords_array = jnp.asarray(coords, dtype=jnp.float32)
    if coords_array.ndim == 2:
        if weights is not None:
            raise ValueError("weights must be None for single-structure input")
        return hdx_forward(coords_array, features, config)
    if coords_array.ndim != 3:
        raise ValueError("coords must have shape (N_atoms, 3) or (T, N_atoms, 3)")

    per_frame = [hdx_forward(coords_array[t], features, config) for t in range(coords_array.shape[0])]
    all_nc = jnp.stack([frame["Nc"] for frame in per_frame], axis=0)
    all_nh = jnp.stack([frame["Nh"] for frame in per_frame], axis=0)
    all_ln = jnp.stack([frame["ln_Pf"] for frame in per_frame], axis=0)

    if weights is None:
        return {
            "Nc": jnp.mean(all_nc, axis=0),
            "Nh": jnp.mean(all_nh, axis=0),
            "ln_Pf": jnp.mean(all_ln, axis=0),
        }

    w = jnp.asarray(weights, dtype=all_ln.dtype)
    if w.ndim != 1 or w.shape[0] != coords_array.shape[0]:
        raise ValueError("weights must have shape (T,) and match trajectory length")
    w = w / jnp.sum(w)
    return {
        "Nc": jnp.sum(w[:, None] * all_nc, axis=0),
        "Nh": jnp.sum(w[:, None] * all_nh, axis=0),
        "ln_Pf": jnp.sum(w[:, None] * all_ln, axis=0),
    }
