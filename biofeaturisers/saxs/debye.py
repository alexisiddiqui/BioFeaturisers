"""Chunked Debye six-partial accumulation for SAXS."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from biofeaturisers.core.pairwise import dist_from_sq_block, dist_matrix_block
from biofeaturisers.core.safe_math import diagonal_self_pairs, safe_sinc

from .features import SAXSFeatures


def six_partial_sums_block(
    sinc_block: Float[Array, "b_i b_j n_q"],
    fv_i: Float[Array, "b_i n_q"],
    fv_j: Float[Array, "b_j n_q"],
    fe_i: Float[Array, "b_i n_q"],
    fe_j: Float[Array, "b_j n_q"],
    fs_i: Float[Array, "b_i n_q"],
    fs_j: Float[Array, "b_j n_q"],
) -> Float[Array, "6 n_q"]:
    """Compute FoXS six partial sums for one sinc-weighted coordinate block."""
    sinc_arr = jnp.asarray(sinc_block, dtype=jnp.float32)
    fv_i_arr = jnp.asarray(fv_i, dtype=jnp.float32)
    fv_j_arr = jnp.asarray(fv_j, dtype=jnp.float32)
    fe_i_arr = jnp.asarray(fe_i, dtype=jnp.float32)
    fe_j_arr = jnp.asarray(fe_j, dtype=jnp.float32)
    fs_i_arr = jnp.asarray(fs_i, dtype=jnp.float32)
    fs_j_arr = jnp.asarray(fs_j, dtype=jnp.float32)

    w_v = jnp.einsum("ijq,jq->iq", sinc_arr, fv_j_arr)
    w_e = jnp.einsum("ijq,jq->iq", sinc_arr, fe_j_arr)
    w_s = jnp.einsum("ijq,jq->iq", sinc_arr, fs_j_arr)

    iaa = jnp.sum(fv_i_arr * w_v, axis=0)
    icc = jnp.sum(fe_i_arr * w_e, axis=0)
    iss = jnp.sum(fs_i_arr * w_s, axis=0)
    iac = jnp.sum(fv_i_arr * w_e + fe_i_arr * w_v, axis=0)
    ias = jnp.sum(fv_i_arr * w_s + fs_i_arr * w_v, axis=0)
    ics = jnp.sum(fe_i_arr * w_s + fs_i_arr * w_e, axis=0)
    return jnp.stack((iaa, icc, iss, iac, ias, ics), axis=0)


def _diagonal_partials(
    ff_vac: Float[Array, "n_sel n_q"],
    ff_excl: Float[Array, "n_sel n_q"],
    ff_water: Float[Array, "n_sel n_q"],
) -> Float[Array, "6 n_q"]:
    return jnp.stack(
        (
            diagonal_self_pairs(ff_vac),
            diagonal_self_pairs(ff_excl),
            diagonal_self_pairs(ff_water),
            2.0 * jnp.sum(ff_vac * ff_excl, axis=0),
            2.0 * jnp.sum(ff_vac * ff_water, axis=0),
            2.0 * jnp.sum(ff_excl * ff_water, axis=0),
        ),
        axis=0,
    )


@partial(jax.jit, static_argnames=("chunk_size",))
def _saxs_six_partials_arrays(
    coords_sel: Float[Array, "n_sel 3"],
    q_values: Float[Array, "n_q"],
    ff_vac: Float[Array, "n_sel n_q"],
    ff_excl: Float[Array, "n_sel n_q"],
    ff_water: Float[Array, "n_sel n_q"],
    chunk_size: int,
) -> Float[Array, "6 n_q"]:
    n_sel = int(coords_sel.shape[0])
    n_q = int(q_values.shape[0])
    pad_n = (-n_sel) % chunk_size

    coords_p = jnp.pad(coords_sel, ((0, pad_n), (0, 0)))
    ff_vac_p = jnp.pad(ff_vac, ((0, pad_n), (0, 0)))
    ff_excl_p = jnp.pad(ff_excl, ((0, pad_n), (0, 0)))
    ff_water_p = jnp.pad(ff_water, ((0, pad_n), (0, 0)))

    n_chunks = coords_p.shape[0] // chunk_size
    coords_chunks = coords_p.reshape((n_chunks, chunk_size, 3))
    ff_vac_chunks = ff_vac_p.reshape((n_chunks, chunk_size, n_q))
    ff_excl_chunks = ff_excl_p.reshape((n_chunks, chunk_size, n_q))
    ff_water_chunks = ff_water_p.reshape((n_chunks, chunk_size, n_q))
    diag_mask = (1.0 - jnp.eye(chunk_size, dtype=jnp.float32))[:, :, None]

    diag_partials = _diagonal_partials(ff_vac=ff_vac, ff_excl=ff_excl, ff_water=ff_water)

    def _outer_scan(carry: jax.Array, i: jax.Array) -> tuple[jax.Array, None]:
        fv_i = ff_vac_chunks[i]
        fe_i = ff_excl_chunks[i]
        fs_i = ff_water_chunks[i]
        ci = coords_chunks[i]

        def _inner_scan(inner_carry: jax.Array, j: jax.Array) -> tuple[jax.Array, None]:
            cj = coords_chunks[j]
            dist_sq = dist_matrix_block(ci, cj)
            dist = dist_from_sq_block(dist_sq)
            qr = q_values[None, None, :] * dist[:, :, None]
            sinc_block = safe_sinc(qr)
            sinc_block = jax.lax.cond(
                i == j,
                lambda value: value * diag_mask,
                lambda value: value,
                sinc_block,
            )
            block = six_partial_sums_block(
                sinc_block=sinc_block,
                fv_i=fv_i,
                fv_j=ff_vac_chunks[j],
                fe_i=fe_i,
                fe_j=ff_excl_chunks[j],
                fs_i=fs_i,
                fs_j=ff_water_chunks[j],
            )
            return inner_carry + block, None

        carry, _ = jax.lax.scan(_inner_scan, carry, jnp.arange(n_chunks))
        return carry, None

    offdiag_partials, _ = jax.lax.scan(
        _outer_scan,
        jnp.zeros((6, n_q), dtype=jnp.float32),
        jnp.arange(n_chunks),
    )
    return diag_partials + offdiag_partials


def saxs_six_partials(
    coords: Float[Array, "n_atoms 3"],
    features: SAXSFeatures,
    chunk_size: int = 512,
) -> Float[Array, "6 n_q"]:
    """Compute FoXS six partial sums without materialising ``(N, N, Q)`` tensors."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    coords_arr = jnp.asarray(coords, dtype=jnp.float32)
    if coords_arr.ndim != 2 or int(coords_arr.shape[1]) != 3:
        raise ValueError(f"coords must have shape (n_atoms, 3), got {tuple(coords_arr.shape)}")

    atom_idx = jnp.asarray(features.atom_idx, dtype=jnp.int32)
    q_values = jnp.asarray(features.q_values, dtype=jnp.float32)
    ff_vac = jnp.asarray(features.ff_vac, dtype=jnp.float32)
    ff_excl = jnp.asarray(features.ff_excl, dtype=jnp.float32)
    ff_water = jnp.asarray(features.ff_water, dtype=jnp.float32)

    if int(atom_idx.shape[0]) == 0:
        return jnp.zeros((6, int(q_values.shape[0])), dtype=jnp.float32)

    coords_sel = coords_arr[atom_idx]
    return _saxs_six_partials_arrays(
        coords_sel=coords_sel,
        q_values=q_values,
        ff_vac=ff_vac,
        ff_excl=ff_excl,
        ff_water=ff_water,
        chunk_size=chunk_size,
    )


__all__ = ["saxs_six_partials", "six_partial_sums_block"]
