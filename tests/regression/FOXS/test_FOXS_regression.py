"""FOXS regression test for SAXS computation.

Tests that biofeaturisers' SAXS profiles match precomputed FoXS reference
using the 5PTI NMR structure (first frame). Validates numerical agreement
to ensure consistency with standard SAXS computation tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from biotite.structure import AtomArray

from biofeaturisers.config import SAXSConfig
from biofeaturisers.saxs.foxs import saxs_forward
from biofeaturisers.saxs.featurise import featurise

if TYPE_CHECKING:
    from typing import Sequence


def test_foxs_5pti_profile_matches_reference(
    pdb_5pti_frame0: AtomArray,
    foxs_5pti_reference_profile: tuple[Sequence[float], Sequence[float], Sequence[float]],
) -> None:
    """Test SAXS profile against FoXS reference for 5PTI.
    
    Validates that biofeaturisers' forward SAXS computation produces
    profiles numerically close to FoXS reference data, using standard
    parameters (c1=1.0, c2=0.0, no hydration adjustments).
    
    Args:
        pdb_5pti_frame0: 5PTI preprocessed structure fixture (global).
        foxs_5pti_reference_profile: FoXS reference (q, I, sigma) tuple.
    """
    atom_array = pdb_5pti_frame0
    q_ref, i_ref, sigma_ref = foxs_5pti_reference_profile
    
    # Convert to arrays
    q_ref_arr = np.asarray(q_ref, dtype=np.float32)
    i_ref_arr = np.asarray(i_ref, dtype=np.float32)
    sigma_ref_arr = np.asarray(sigma_ref, dtype=np.float32)
    
    # Compute SAXS profile with matching config
    config = SAXSConfig(
        q_min=float(q_ref_arr[0]),
        q_max=float(q_ref_arr[-1]),
        n_q=int(len(q_ref_arr)),
        chunk_size=256,
        fit_c1_c2=False,
        c1=1.0,
        c2=0.0,
    )
    
    # Compute features and forward model
    features = featurise(atom_array=atom_array, config=config)
    coords = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))
    i_q = saxs_forward(coords=coords, features=features, c1=1.0, c2=0.0)
    i_q_arr = np.asarray(i_q)
    
    # Validate shape and values
    assert i_q_arr.shape == i_ref_arr.shape, (
        f"Profile shape mismatch: computed {i_q_arr.shape}, "
        f"reference {i_ref_arr.shape}"
    )
    assert np.isfinite(i_q_arr).all(), "Computed profile contains non-finite values"
    
    # Compare with numerical tolerance
    # Use absolute and relative tolerance to handle varying intensity scales
    rtol = 0.0005  # 0.05% relative tolerance
    atol = 0.0001 * np.mean(i_ref_arr)  # 0.01% of mean intensity as absolute tolerance

    try:
        np.testing.assert_allclose(i_q_arr, i_ref_arr, rtol=rtol, atol=atol)
    except AssertionError as e:
        # Provide more diagnostic information on failure
        max_rel_error = np.max(np.abs(i_q_arr - i_ref_arr) / (np.abs(i_ref_arr) + 1e-10))
        mean_rel_error = np.mean(np.abs(i_q_arr - i_ref_arr) / (np.abs(i_ref_arr) + 1e-10))
        pytest.fail(
            f"SAXS profile mismatch exceeds tolerance:\n"
            f"  rtol={rtol}, atol={atol:.2f}\n"
            f"  max relative error: {max_rel_error:.4f}\n"
            f"  mean relative error: {mean_rel_error:.4f}\n"
            f"  {str(e)}"
        )