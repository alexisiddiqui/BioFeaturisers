from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .topology import MinimalTopology

@dataclass
class OutputIndex:
    """Atom-level boolean mask + derived residue mask.
    Applies to both probe set and environment set selection.
    """
    # Selection applied to the full atom array (n_atoms_total)
    atom_mask:    np.ndarray   # (N,) bool — atoms included in environment
    probe_mask:   np.ndarray   # (N,) bool — atoms that ARE probes (subset of atom_mask)
    output_mask:  np.ndarray   # (R,) bool — residues appearing in output arrays

    # Integer index arrays (JAX-ready, pre-gathered)
    atom_idx:     np.ndarray   # (n_env,)   int32 — environment atom indices
    probe_idx:    np.ndarray   # (n_probe,) int32 — probe atom indices
    output_res_idx: np.ndarray # (n_out,)   int32 — output residue indices

    @classmethod
    def from_selection(
        cls,
        topology: MinimalTopology,
        include_chains: Optional[list[str]] = None,   # None = all chains
        exclude_chains: Optional[list[str]] = None,
        include_hetatm: bool = False,                 # ligands in environment
        custom_atom_mask: Optional[np.ndarray] = None # override all of the above
    ) -> "OutputIndex":
        num_atoms = topology.atom_names.shape[0]
        num_res = topology.res_unique_ids.shape[0]

        if custom_atom_mask is None:
            atom_mask = np.ones(num_atoms, dtype=bool)
        else:
            atom_mask = np.asarray(custom_atom_mask, dtype=bool)
            if atom_mask.shape != (num_atoms,):
                raise ValueError("custom_atom_mask must have shape (n_atoms,)")

        if include_chains is not None:
            atom_mask &= np.isin(topology.chain_ids, np.asarray(include_chains, dtype=str))
        if exclude_chains is not None:
            atom_mask &= ~np.isin(topology.chain_ids, np.asarray(exclude_chains, dtype=str))
        if not include_hetatm:
            atom_mask &= ~topology.is_hetatm

        # Probes are always non-HETATM atoms.
        probe_mask = atom_mask & (~topology.is_hetatm)

        atom_res_keys = topology.atom_res_keys
        output_mask = np.zeros(num_res, dtype=bool)
        if np.any(probe_mask):
            probe_keys = set(atom_res_keys[probe_mask].tolist())
            output_mask = np.array([key in probe_keys for key in topology.res_unique_ids], dtype=bool)

        return cls(
            atom_mask=atom_mask,
            probe_mask=probe_mask,
            output_mask=output_mask,
            atom_idx=np.flatnonzero(atom_mask).astype(np.int32),
            probe_idx=np.flatnonzero(probe_mask).astype(np.int32),
            output_res_idx=np.flatnonzero(output_mask).astype(np.int32),
        )
