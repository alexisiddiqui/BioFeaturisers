"""Fixtures for HDX BV model regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict

import pytest
import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure import AtomArrayStack


@pytest.fixture
def bpti_experimental_data() -> Dict[int, float]:
    """Load experimental BPTI HDX exchange rates (log10 Pf values).
    
    Returns:
        dict: Mapping from residue ID to log10(Pf) experimental value (as stored in file).
              The test converts these to natural log to match model output.
    """
    ref_path = Path(__file__).resolve().parent.parent.parent / "BPTI_expt_Pfs.dat"
    data = np.loadtxt(ref_path, comments="#")
    
    # data columns: ResID, log10(PF)
    res_ids = data[:, 0].astype(int)
    log_pf_values = data[:, 1].astype(np.float32)
    
    return dict(zip(res_ids, log_pf_values))


@pytest.fixture
def pdb_5pti_ensemble() -> AtomArrayStack:
    """Load 5PTI NMR ensemble (20 frames) with realistic coordinate variations.
    
    Creates an ensemble of 20 frames from the 5PTI structure with small Gaussian
    perturbations (sigma=0.5 Å) to simulate NMR-like structural uncertainty.
    The preprocessed structure has solvent removed and D→H normalized.
    
    Returns:
        AtomArrayStack: 20 frames of 5PTI structure (1,087 atoms per frame).
    """
    fixture_path = Path(__file__).resolve().parent.parent.parent / "regression" / "FOXS" / "fixtures" / "5PTI_frame0.pdb"
    pdb = PDBFile.read(fixture_path)
    single_frame = pdb.get_structure(model=1)
    
    # Create ensemble with coordinate variations
    n_atoms = len(single_frame)
    n_frames = 20
    ensemble = AtomArrayStack(depth=n_frames, length=n_atoms)
    
    # Copy annotations from single frame
    ensemble.atom_name = single_frame.atom_name
    ensemble.res_name = single_frame.res_name
    ensemble.res_id = single_frame.res_id
    ensemble.chain_id = single_frame.chain_id
    ensemble.element = single_frame.element
    ensemble.hetero = single_frame.hetero
    ensemble.ins_code = single_frame.ins_code
    
    # Add coordinates with small Gaussian perturbations (NMR-like uncertainty)
    np.random.seed(42)  # Deterministic for reproducible tests
    for i in range(n_frames):
        noise = np.random.normal(0, 0.5, single_frame.coord.shape)
        ensemble.coord[i] = single_frame.coord + noise
    
    return ensemble
