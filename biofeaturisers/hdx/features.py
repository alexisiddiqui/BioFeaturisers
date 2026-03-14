"""HDX feature dataclass contracts and persistence."""

from __future__ import annotations

from dataclasses import dataclass

import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology
from biofeaturisers.io.load import load_feature_bundle, output_index_from_arrays
from biofeaturisers.io.save import output_index_arrays, save_feature_bundle


def _as_str_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=str)


@dataclass(slots=True)
class HDXFeatures:
    """Static HDX featurisation contract."""

    topology: MinimalTopology
    output_index: OutputIndex
    amide_N_idx: Int[Array, "n_probe"]
    amide_H_idx: Int[Array, "n_probe"]
    heavy_atom_idx: Int[Array, "n_heavy"]
    backbone_O_idx: Int[Array, "n_bb_o"]
    excl_mask_c: Float[Array, "n_probe n_heavy"]
    excl_mask_h: Float[Array, "n_probe n_bb_o"]
    res_keys: np.ndarray
    res_names: np.ndarray
    can_exchange: Bool[Array, "n_out"]
    kint: Float[Array, "n_out"] | None = None
    amide_CA_idx: Int[Array, "n_probe"] | None = None
    amide_prev_C_idx: Int[Array, "n_probe"] | None = None

    def __post_init__(self) -> None:
        self.amide_N_idx = jnp.asarray(self.amide_N_idx, dtype=jnp.int32)
        self.amide_H_idx = jnp.asarray(self.amide_H_idx, dtype=jnp.int32)
        self.heavy_atom_idx = jnp.asarray(self.heavy_atom_idx, dtype=jnp.int32)
        self.backbone_O_idx = jnp.asarray(self.backbone_O_idx, dtype=jnp.int32)
        self.excl_mask_c = jnp.asarray(self.excl_mask_c, dtype=jnp.float32)
        self.excl_mask_h = jnp.asarray(self.excl_mask_h, dtype=jnp.float32)
        self.can_exchange = jnp.asarray(self.can_exchange, dtype=jnp.bool_)
        if self.kint is not None:
            self.kint = jnp.asarray(self.kint, dtype=jnp.float32)
        if self.amide_CA_idx is not None:
            self.amide_CA_idx = jnp.asarray(self.amide_CA_idx, dtype=jnp.int32)
        if self.amide_prev_C_idx is not None:
            self.amide_prev_C_idx = jnp.asarray(self.amide_prev_C_idx, dtype=jnp.int32)

        self.res_keys = _as_str_array(self.res_keys)
        self.res_names = _as_str_array(self.res_names)

        chex.assert_rank(self.amide_N_idx, 1)
        chex.assert_rank(self.amide_H_idx, 1)
        chex.assert_rank(self.heavy_atom_idx, 1)
        chex.assert_rank(self.backbone_O_idx, 1)
        chex.assert_rank(self.excl_mask_c, 2)
        chex.assert_rank(self.excl_mask_h, 2)
        chex.assert_rank(self.can_exchange, 1)

        n_probe = int(self.amide_N_idx.shape[0])
        if int(self.amide_H_idx.shape[0]) != n_probe:
            raise ValueError("amide_N_idx and amide_H_idx must have the same length")
        if int(self.excl_mask_c.shape[0]) != n_probe:
            raise ValueError("excl_mask_c row count must match probe count")
        if int(self.excl_mask_h.shape[0]) != n_probe:
            raise ValueError("excl_mask_h row count must match probe count")
        if int(self.excl_mask_c.shape[1]) != int(self.heavy_atom_idx.shape[0]):
            raise ValueError("excl_mask_c column count must match heavy_atom_idx length")
        if int(self.excl_mask_h.shape[1]) != int(self.backbone_O_idx.shape[0]):
            raise ValueError("excl_mask_h column count must match backbone_O_idx length")
        if (self.amide_CA_idx is None) != (self.amide_prev_C_idx is None):
            raise ValueError(
                "amide_CA_idx and amide_prev_C_idx must both be provided or both be None"
            )
        if self.amide_CA_idx is not None:
            chex.assert_rank(self.amide_CA_idx, 1)
            chex.assert_rank(self.amide_prev_C_idx, 1)
            if int(self.amide_CA_idx.shape[0]) != n_probe:
                raise ValueError("amide_CA_idx length must match probe count")
            if int(self.amide_prev_C_idx.shape[0]) != n_probe:
                raise ValueError("amide_prev_C_idx length must match probe count")
        if bool(jnp.any(self.amide_H_idx < 0)) and self.amide_CA_idx is None:
            raise ValueError(
                "amide_CA_idx/amide_prev_C_idx are required when amide_H_idx contains -1"
            )

        n_out = int(self.res_keys.shape[0])
        if int(self.res_names.shape[0]) != n_out or int(self.can_exchange.shape[0]) != n_out:
            raise ValueError("res_keys/res_names/can_exchange lengths must match")
        if int(self.output_index.output_res_idx.shape[0]) != n_out:
            raise ValueError("output_res_idx length must match HDX output residue metadata length")
        if self.kint is not None and int(self.kint.shape[0]) != n_out:
            raise ValueError("kint length must match output residue length")

    def save(self, prefix: str) -> None:
        """Persist static HDX features as NPZ + topology JSON."""
        save_feature_bundle(
            prefix=prefix,
            topology=self.topology,
            arrays={
                "amide_N_idx": np.asarray(self.amide_N_idx, dtype=np.int32),
                "amide_H_idx": np.asarray(self.amide_H_idx, dtype=np.int32),
                "heavy_atom_idx": np.asarray(self.heavy_atom_idx, dtype=np.int32),
                "backbone_O_idx": np.asarray(self.backbone_O_idx, dtype=np.int32),
                "excl_mask_c": np.asarray(self.excl_mask_c, dtype=np.float32),
                "excl_mask_h": np.asarray(self.excl_mask_h, dtype=np.float32),
                "res_keys": self.res_keys,
                "res_names": self.res_names,
                "can_exchange": np.asarray(self.can_exchange, dtype=bool),
                **output_index_arrays(self.output_index),
                "has_kint": np.asarray(self.kint is not None, dtype=bool),
                "kint": (
                    np.asarray(self.kint, dtype=np.float32)
                    if self.kint is not None
                    else np.asarray([], dtype=np.float32)
                ),
                "has_amide_geom": np.asarray(self.amide_CA_idx is not None, dtype=bool),
                "amide_CA_idx": (
                    np.asarray(self.amide_CA_idx, dtype=np.int32)
                    if self.amide_CA_idx is not None
                    else np.asarray([], dtype=np.int32)
                ),
                "amide_prev_C_idx": (
                    np.asarray(self.amide_prev_C_idx, dtype=np.int32)
                    if self.amide_prev_C_idx is not None
                    else np.asarray([], dtype=np.int32)
                ),
            },
        )

    @classmethod
    def load(cls, prefix: str) -> "HDXFeatures":
        """Load HDX features from NPZ + topology JSON."""
        topology, data = load_feature_bundle(prefix)
        output_index = output_index_from_arrays(data)
        has_kint = bool(np.asarray(data["has_kint"]).item())
        kint = data["kint"] if has_kint else None
        has_amide_geom = bool(np.asarray(data["has_amide_geom"]).item()) if "has_amide_geom" in data else False
        amide_ca_idx = data["amide_CA_idx"] if has_amide_geom else None
        amide_prev_c_idx = data["amide_prev_C_idx"] if has_amide_geom else None

        return cls(
            topology=topology,
            output_index=output_index,
            amide_N_idx=data["amide_N_idx"],
            amide_H_idx=data["amide_H_idx"],
            heavy_atom_idx=data["heavy_atom_idx"],
            backbone_O_idx=data["backbone_O_idx"],
            excl_mask_c=data["excl_mask_c"],
            excl_mask_h=data["excl_mask_h"],
            res_keys=data["res_keys"],
            res_names=data["res_names"],
            can_exchange=data["can_exchange"],
            kint=kint,
            amide_CA_idx=amide_ca_idx,
            amide_prev_C_idx=amide_prev_c_idx,
        )
