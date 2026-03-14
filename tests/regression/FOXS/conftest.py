"""Fixtures for FOXS regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

import pytest


@pytest.fixture
def foxs_5pti_reference_profile() -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
    """Load precomputed FoXS reference profile for 5PTI.
    
    Returns:
        tuple: (q_values, intensities, uncertainties) arrays from FoXS.
    """
    import numpy as np
    
    ref_path = Path(__file__).resolve().parent / "fixtures" / "5PTI_frame0_foxs_reference.dat"
    data = np.loadtxt(ref_path, comments="#")
    
    # data columns: q, I, sigma
    q_vals = data[:, 0]
    i_vals = data[:, 1]
    sigma_vals = data[:, 2]
    
    return q_vals, i_vals, sigma_vals

