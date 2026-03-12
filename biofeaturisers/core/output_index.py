from typing import List, Optional
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
        include_chains: Optional[List[str]] = None,   # None = all chains
        exclude_chains: Optional[List[str]] = None,
        include_hetatm: bool = False,              # ligands in environment
        custom_atom_mask: Optional[np.ndarray] = None # override all of the above
    ) -> "OutputIndex":
        """
        Implementation stub for constructing OutputIndex from topology and selection criteria.
        Full logic will involve chain/residue filtering and mapping.
        """
        # For now, return a dummy full selection
        num_atoms = len(topology.atom_names)
        num_res = len(topology.res_unique_ids)
        
        atom_mask = np.ones(num_atoms, dtype=bool)
        if custom_atom_mask is not None:
            atom_mask = custom_atom_mask
            
        return cls(
            atom_mask=atom_mask,
            probe_mask=atom_mask.copy(),
            output_mask=np.ones(num_res, dtype=bool),
            atom_idx=np.where(atom_mask)[0].astype(np.int32),
            probe_idx=np.where(atom_mask)[0].astype(np.int32),
            output_res_idx=np.arange(num_res, dtype=np.int32)
        )
