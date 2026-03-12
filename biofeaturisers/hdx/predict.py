from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ..config import HDXConfig
from ..core.output_index import OutputIndex
from .featurise import extract_coords, featurise
from .forward import forward
from .hdxrate import predict_uptake


def predict(
    structure: Any,
    config: HDXConfig | None = None,
    output_index: OutputIndex | None = None,
    weights=None,
    peptide_mask=None,
):
    cfg = config if config is not None else HDXConfig()
    features = featurise(structure=structure, config=cfg, output_index=output_index)
    coords = extract_coords(structure)
    result = forward(coords=coords, features=features, config=cfg, weights=weights)

    if cfg.use_hdxrate and peptide_mask is not None and len(cfg.timepoints) > 0:
        if features.kint is None:
            raise ValueError("kint is not available. Re-run featurise with use_hdxrate=True.")
        result["uptake"] = predict_uptake(
            ln_Pf=result["ln_Pf"],
            kint=jnp.asarray(features.kint, dtype=jnp.float32),
            can_exchange=jnp.asarray(features.can_exchange.astype(np.float32)),
            peptide_mask=jnp.asarray(peptide_mask, dtype=jnp.float32),
            timepoints=tuple(cfg.timepoints),
        )

    return result
