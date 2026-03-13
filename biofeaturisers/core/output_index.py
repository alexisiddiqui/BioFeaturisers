"""Output routing masks and index arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology import MinimalTopology


@dataclass(slots=True)
class OutputIndex:
    """Selection state for atom/probe/residue routing."""

    atom_mask: np.ndarray
    probe_mask: np.ndarray
    output_mask: np.ndarray
    atom_idx: np.ndarray
    probe_idx: np.ndarray
    output_res_idx: np.ndarray

    def __post_init__(self) -> None:
        if self.atom_mask.dtype != bool:
            raise ValueError("atom_mask must be boolean")
        if self.probe_mask.dtype != bool:
            raise ValueError("probe_mask must be boolean")
        if self.output_mask.dtype != bool:
            raise ValueError("output_mask must be boolean")
        if self.atom_mask.shape != self.probe_mask.shape:
            raise ValueError("probe_mask shape must match atom_mask shape")
        if np.any(self.probe_mask & ~self.atom_mask):
            raise ValueError("probe_mask must be a subset of atom_mask")

    @classmethod
    def from_selection(
        cls,
        topology: MinimalTopology,
        include_chains: list[str] | None = None,
        exclude_chains: list[str] | None = None,
        include_hetatm: bool = False,
        custom_atom_mask: np.ndarray | None = None,
    ) -> "OutputIndex":
        n_atoms = topology.atom_names.shape[0]
        if custom_atom_mask is not None:
            atom_mask = np.asarray(custom_atom_mask, dtype=bool)
            if atom_mask.shape != (n_atoms,):
                raise ValueError(
                    f"custom_atom_mask shape must be {(n_atoms,)}, got {atom_mask.shape}"
                )
        else:
            atom_mask = np.ones(n_atoms, dtype=bool)
            if include_chains is not None:
                atom_mask &= np.isin(topology.chain_ids, np.asarray(include_chains, dtype=str))
            if exclude_chains is not None:
                atom_mask &= ~np.isin(topology.chain_ids, np.asarray(exclude_chains, dtype=str))
            if not include_hetatm:
                atom_mask &= ~topology.is_hetatm

        probe_mask = atom_mask & (~topology.is_hetatm)

        selected_res_keys = {
            f"{chain}:{res_id}"
            for chain, res_id, selected, het in zip(
                topology.chain_ids, topology.res_ids, atom_mask, topology.is_hetatm, strict=True
            )
            if selected and not het
        }
        output_mask = np.asarray(
            [res_key in selected_res_keys for res_key in topology.res_unique_ids], dtype=bool
        )

        atom_idx = np.flatnonzero(atom_mask).astype(np.int32)
        probe_idx = np.flatnonzero(probe_mask).astype(np.int32)
        output_res_idx = np.flatnonzero(output_mask).astype(np.int32)

        return cls(
            atom_mask=atom_mask,
            probe_mask=probe_mask,
            output_mask=output_mask,
            atom_idx=atom_idx,
            probe_idx=probe_idx,
            output_res_idx=output_res_idx,
        )

