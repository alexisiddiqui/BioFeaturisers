"""End-to-end HDX prediction wrapper."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from biotite.structure import AtomArray
from jaxtyping import Array, Float

from biofeaturisers.config import HDXConfig
from biofeaturisers.core.ensemble import apply_forward
from biofeaturisers.core.output_index import OutputIndex

from .featurise import featurise
from .forward import hdx_forward
from .hdxrate import predict_uptake


def _stack_forward(frame_coords: Array, config: HDXConfig, features) -> Array:
    result = hdx_forward(frame_coords, features=features, config=config)
    return jnp.stack((result["Nc"], result["Nh"], result["ln_Pf"]), axis=0)


def predict(
    atom_array: AtomArray,
    config: HDXConfig | None = None,
    output_index: OutputIndex | None = None,
    coords: Float[Array, "n_atoms 3"] | Float[Array, "n_frames n_atoms 3"] | None = None,
    weights: Float[Array, "n_frames"] | None = None,
    peptide_mask: Float[Array, "n_peptides n_probe"] | None = None,
) -> dict[str, Array]:
    """Run `featurise -> forward` and optionally HDXrate uptake prediction."""
    cfg = config or HDXConfig()
    features = featurise(atom_array=atom_array, config=cfg, output_index=output_index)

    if coords is None:
        coords_arr = jnp.asarray(np.asarray(atom_array.coord, dtype=np.float32))
    else:
        coords_arr = jnp.asarray(coords, dtype=jnp.float32)

    if coords_arr.ndim == 2:
        result = hdx_forward(coords_arr, features=features, config=cfg)
    elif coords_arr.ndim == 3:
        stacked = apply_forward(
            lambda frame: _stack_forward(frame, config=cfg, features=features),
            coords=coords_arr,
            weights=weights,
            batch_size=cfg.batch_size,
        )
        result = {"Nc": stacked[0], "Nh": stacked[1], "ln_Pf": stacked[2]}
    else:
        raise ValueError(
            f"coords must have shape (n_atoms, 3) or (n_frames, n_atoms, 3), got {coords_arr.shape}"
        )

    if cfg.use_hdxrate:
        if features.kint is None:
            raise ValueError("features.kint is required when use_hdxrate=True")
        if len(cfg.timepoints) == 0:
            raise ValueError("HDXConfig.timepoints must be non-empty when use_hdxrate=True")

        can_exchange = jnp.asarray(features.can_exchange, dtype=jnp.bool_)
        probe_mask = np.asarray(can_exchange, dtype=bool)
        ln_pf = jnp.asarray(result["ln_Pf"], dtype=jnp.float32)

        if int(np.sum(probe_mask)) != int(ln_pf.shape[0]):
            raise ValueError(
                "Mismatch between probe count and exchangeable residue count in feature metadata"
            )

        kint_probe = jnp.asarray(features.kint, dtype=jnp.float32)[probe_mask]
        can_exchange_probe = jnp.asarray(can_exchange, dtype=jnp.float32)[probe_mask]
        if peptide_mask is None:
            peptide_mask_arr = jnp.eye(int(ln_pf.shape[0]), dtype=jnp.float32)
        else:
            peptide_mask_arr = jnp.asarray(peptide_mask, dtype=jnp.float32)
            if peptide_mask_arr.ndim != 2 or int(peptide_mask_arr.shape[1]) != int(ln_pf.shape[0]):
                raise ValueError("peptide_mask must have shape (n_peptides, n_probe)")

        result["uptake"] = predict_uptake(
            ln_pf=ln_pf,
            kint=kint_probe,
            can_exchange=can_exchange_probe,
            peptide_mask=peptide_mask_arr,
            timepoints=tuple(float(value) for value in cfg.timepoints),
        )

    return result


__all__ = ["predict"]

