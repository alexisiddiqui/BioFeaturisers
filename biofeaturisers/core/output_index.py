"""Output routing masks and index arrays."""

from __future__ import annotations

from dataclasses import dataclass

import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int

from .topology import MinimalTopology


@dataclass(slots=True)
class OutputIndex:
    """Selection state for atom/probe/residue routing."""

    atom_mask: Bool[Array, "n_atoms"]
    probe_mask: Bool[Array, "n_atoms"]
    output_mask: Bool[Array, "n_residues"]
    atom_idx: Int[Array, "n_env"]
    probe_idx: Int[Array, "n_probe"]
    output_res_idx: Int[Array, "n_out"]

    def __post_init__(self) -> None:
        self.atom_mask = jnp.asarray(self.atom_mask, dtype=jnp.bool_)
        self.probe_mask = jnp.asarray(self.probe_mask, dtype=jnp.bool_)
        self.output_mask = jnp.asarray(self.output_mask, dtype=jnp.bool_)
        self.atom_idx = jnp.asarray(self.atom_idx, dtype=jnp.int32)
        self.probe_idx = jnp.asarray(self.probe_idx, dtype=jnp.int32)
        self.output_res_idx = jnp.asarray(self.output_res_idx, dtype=jnp.int32)

        chex.assert_rank(self.atom_mask, 1)
        chex.assert_rank(self.probe_mask, 1)
        chex.assert_rank(self.output_mask, 1)
        chex.assert_rank(self.atom_idx, 1)
        chex.assert_rank(self.probe_idx, 1)
        chex.assert_rank(self.output_res_idx, 1)

        if self.atom_mask.dtype != jnp.bool_:
            raise ValueError("atom_mask must be boolean")
        if self.probe_mask.dtype != jnp.bool_:
            raise ValueError("probe_mask must be boolean")
        if self.output_mask.dtype != jnp.bool_:
            raise ValueError("output_mask must be boolean")
        if self.atom_idx.dtype != jnp.int32:
            raise ValueError("atom_idx must be int32")
        if self.probe_idx.dtype != jnp.int32:
            raise ValueError("probe_idx must be int32")
        if self.output_res_idx.dtype != jnp.int32:
            raise ValueError("output_res_idx must be int32")
        if self.atom_mask.shape != self.probe_mask.shape:
            raise ValueError("probe_mask shape must match atom_mask shape")
        if bool(jnp.any(self.probe_mask & ~self.atom_mask)):
            raise ValueError("probe_mask must be a subset of atom_mask")

    @classmethod
    def from_selection(
        cls,
        topology: MinimalTopology,
        include_chains: list[str] | None = None,
        exclude_chains: list[str] | None = None,
        include_hetatm: bool = False,
        custom_atom_mask: Bool[Array, "n_atoms"] | np.ndarray | None = None,
    ) -> "OutputIndex":
        n_atoms = topology.atom_names.shape[0]
        is_hetatm_np = np.asarray(topology.is_hetatm, dtype=bool)
        res_ids_np = np.asarray(topology.res_ids, dtype=np.int32)

        if custom_atom_mask is not None:
            atom_mask_np = np.asarray(custom_atom_mask, dtype=bool)
            if atom_mask_np.shape != (n_atoms,):
                raise ValueError(
                    f"custom_atom_mask shape must be {(n_atoms,)}, got {atom_mask_np.shape}"
                )
        else:
            atom_mask_np = np.ones(n_atoms, dtype=bool)
            if include_chains is not None:
                atom_mask_np &= np.isin(topology.chain_ids, np.asarray(include_chains, dtype=str))
            if exclude_chains is not None:
                atom_mask_np &= ~np.isin(topology.chain_ids, np.asarray(exclude_chains, dtype=str))
            if not include_hetatm:
                atom_mask_np &= ~is_hetatm_np

        probe_mask_np = atom_mask_np & (~is_hetatm_np)

        selected_res_keys = {
            f"{chain}:{res_id}"
            for chain, res_id, selected, het in zip(
                topology.chain_ids, res_ids_np, atom_mask_np, is_hetatm_np, strict=True
            )
            if selected and not het
        }
        output_mask_np = np.asarray(
            [res_key in selected_res_keys for res_key in topology.res_unique_ids], dtype=bool
        )

        atom_idx_np = np.flatnonzero(atom_mask_np).astype(np.int32)
        probe_idx_np = np.flatnonzero(probe_mask_np).astype(np.int32)
        output_res_idx_np = np.flatnonzero(output_mask_np).astype(np.int32)

        return cls(
            atom_mask=jnp.asarray(atom_mask_np, dtype=jnp.bool_),
            probe_mask=jnp.asarray(probe_mask_np, dtype=jnp.bool_),
            output_mask=jnp.asarray(output_mask_np, dtype=jnp.bool_),
            atom_idx=jnp.asarray(atom_idx_np, dtype=jnp.int32),
            probe_idx=jnp.asarray(probe_idx_np, dtype=jnp.int32),
            output_res_idx=jnp.asarray(output_res_idx_np, dtype=jnp.int32),
        )

