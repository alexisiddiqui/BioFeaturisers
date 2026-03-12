from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class HDXFeatures:
    """
    Features required for HDX forward computation.
    """
    amide_coords: np.ndarray # (N_amides, 3)
    exclusion_mask: np.ndarray # (N_amides, N_heavy_atoms) boolean or int mask
    heavy_atom_coords: np.ndarray # (N_heavy_atoms, 3)
    k_int: np.ndarray # (N_amides,) specific intrinsic rate per amide
    # Any other structural features

@dataclass
class SAXSFeatures:
    """
    Features required for SAXS computation.
    """
    coords: np.ndarray # (N_atoms, 3)
    form_factors_vac: np.ndarray # (N_atoms, num_q_points)
    form_factors_excl: np.ndarray # (N_atoms, num_q_points)
    static_sasa: np.ndarray # (N_atoms,) Shrake-Rupley SASA
    # Any other features required for SAXS
