"""BV model regression test for HDX protection factor prediction.

Tests that biofeaturisers' HDX predictions using the Bai-Vugmeyster (BV) model
match experimental BPTI exchange rates using rank correlation. The test featurizes
the 5PTI NMR structure and validates that predicted vs. experimental protection
factors have strong Spearman rank correlation (rho > 0.7, p < 0.01), indicating
the model captures relative exchange rate ordering even if absolute values differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from biotite.structure import AtomArray
from scipy.stats import spearmanr

from biofeaturisers.config import HDXConfig
from biofeaturisers.hdx.predict import predict

if TYPE_CHECKING:
    from typing import Dict


def test_bv_model_5pti_predictions_match_bpti_experimental(
    pdb_5pti_ensemble,
    bpti_experimental_data: Dict[int, float],
) -> None:
    """Test HDX predictions against BPTI experimental exchange rates.
    
    Validates that biofeaturisers' BV model protection factor predictions
    have strong rank correlation with experimental BPTI data when averaged
    over the NMR ensemble.
    
    Args:
        pdb_5pti_ensemble: 5PTI NMR ensemble (20 frames) from fixture.
        bpti_experimental_data: Dict mapping residue ID to experimental log10(Pf).
    """
    # Use the first frame to get the topology
    atom_array = pdb_5pti_ensemble[0]
    ensemble_coords = np.asarray(pdb_5pti_ensemble.coord, dtype=np.float32)
    
    # Create HDX config with default parameters
    config = HDXConfig()
    
    # Run forward model on ensemble coordinates (returns frame-averaged predictions)
    result = predict(atom_array, config=config, coords=ensemble_coords)
    ln_pf_computed = np.asarray(result["ln_Pf"], dtype=np.float32)
    
    # Get features to map output predictions back to residue IDs
    from biofeaturisers.hdx.featurise import featurise
    features = featurise(atom_array=atom_array, config=config)
    can_exchange = np.asarray(features.can_exchange, dtype=bool)
    res_keys = np.asarray(features.res_keys, dtype=str)
    
    # Build mapping from residue ID to computed ln_pf (only exchangeable residues)
    computed_map = {}
    output_idx = 0
    for res_key_idx, can_ex in enumerate(can_exchange):
        if can_ex:
            res_key = res_keys[res_key_idx]
            res_id = int(res_key.split(":")[1])
            computed_map[res_id] = ln_pf_computed[output_idx]
            output_idx += 1
    
    # Extract predictions for residues in experimental data
    exp_res_ids = np.array(list(bpti_experimental_data.keys()), dtype=int)
    exp_log10_pf_values = np.array([bpti_experimental_data[rid] for rid in exp_res_ids], dtype=np.float32)
    
    # Convert experimental log10(Pf) to natural log to match model output
    exp_log_pf_values = exp_log10_pf_values * np.log(10.0)
    
    # Find matching residues between computed and experimental
    matched_computed = []
    matched_experimental = []
    
    for i, rid in enumerate(exp_res_ids):
        if rid in computed_map:
            matched_computed.append(computed_map[rid])
            matched_experimental.append(exp_log_pf_values[i])
    
    if not matched_computed:
        pytest.fail("No matching residues found between computed predictions and experimental data")
    
    ln_pf_computed_matched = np.array(matched_computed, dtype=np.float32)
    ln_pf_ref_matched = np.array(matched_experimental, dtype=np.float32)
    
    # Validate shape and finite values
    assert ln_pf_computed_matched.shape == ln_pf_ref_matched.shape, (
        f"Shape mismatch: computed {ln_pf_computed_matched.shape}, "
        f"reference {ln_pf_ref_matched.shape}"
    )
    assert np.isfinite(ln_pf_computed_matched).all(), (
        "Computed log(Pf) contains non-finite values"
    )
    assert np.isfinite(ln_pf_ref_matched).all(), (
        "Reference log(Pf) contains non-finite values"
    )
    
    # Compute Spearman rank correlation between predictions and experimental values
    # Spearman correlation is robust to absolute scaling differences and validates
    # that the model captures the relative ordering of exchange rates
    corr, p_value = spearmanr(ln_pf_computed_matched, ln_pf_ref_matched)
    
    # Require strong rank correlation (rho > 0.7) with high statistical significance
    min_corr = 0.8
    max_pval = 0.01
    
    if corr < min_corr or p_value > max_pval:
        pytest.fail(
            f"HDX predictions lack sufficient rank correlation with experiment:\n"
            f"  Spearman rho={corr:.4f} (required > {min_corr})\n"
            f"  p-value={p_value:.4e} (required < {max_pval})\n"
            f"  matched residues: {len(matched_computed)}\n"
            f"  This suggests systematic bias in the BV model for these conditions"
        )
