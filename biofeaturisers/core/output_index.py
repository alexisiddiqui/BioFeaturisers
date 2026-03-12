from dataclasses import dataclass
import numpy as np

@dataclass
class OutputIndex:
    """
    Used for atom/probe/residue masking and selection routing.
    Contains indices that map full system arrays to specific observable subsets.
    """
    atom_indices: np.ndarray
    probe_indices: np.ndarray
    residue_indices: np.ndarray
    mapping_matrix: np.ndarray # Could be a sparse or dense matrix mapping atoms to residues/probes
