"""SAXS featurisation from Biotite topology objects."""

from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray, sasa
from biotite.structure.info import vdw_radius_single

from biofeaturisers.config import SAXSConfig
from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology

from .features import SAXSFeatures
from .form_factors import (
    atomic_volumes_from_elements,
    compute_ff_excl,
    wk_vacuum_form_factors,
    wk_water_form_factor,
)

_SASA_PROBE_RADIUS = 1.4


def _solvent_accessibility_fraction(elements: np.ndarray, atom_sasa: np.ndarray) -> np.ndarray:
    if atom_sasa.shape[0] != elements.shape[0]:
        raise ValueError("atom_sasa and elements must have matching lengths")

    max_sasa = np.zeros((elements.shape[0],), dtype=np.float32)
    for idx, symbol in enumerate(elements):
        radius = vdw_radius_single(str(symbol))
        if radius is None:
            raise ValueError(f"Missing VdW radius for element '{symbol}'")
        max_sasa[idx] = float(4.0 * np.pi * (float(radius) + _SASA_PROBE_RADIUS) ** 2)
    return np.clip(atom_sasa / np.maximum(max_sasa, 1e-8), 0.0, 1.0).astype(np.float32)


def featurise(
    atom_array: AtomArray,
    config: SAXSConfig | None = None,
    output_index: OutputIndex | None = None,
    include_chains: list[str] | None = None,
    exclude_chains: list[str] | None = None,
) -> SAXSFeatures:
    """Create :class:`SAXSFeatures` from a Biotite ``AtomArray``."""
    if not isinstance(atom_array, AtomArray):
        raise TypeError("featurise expects a biotite.structure.AtomArray input")

    cfg = config or SAXSConfig()
    if cfg.ff_table != "waasmaier_kirfel":
        raise NotImplementedError(
            f"SAXS ff_table '{cfg.ff_table}' is not implemented yet; use 'waasmaier_kirfel'"
        )

    topology = MinimalTopology.from_biotite(atom_array)
    include = cfg.include_chains if include_chains is None else include_chains
    exclude = cfg.exclude_chains if exclude_chains is None else exclude_chains

    if output_index is None:
        output_index = OutputIndex.from_selection(
            topology,
            include_chains=include,
            exclude_chains=exclude,
            include_hetatm=cfg.include_hetatm,
        )

    atom_idx = np.asarray(output_index.atom_idx, dtype=np.int32)
    q_values = np.asarray(
        np.linspace(cfg.q_min, cfg.q_max, cfg.n_q, dtype=np.float32),
        dtype=np.float32,
    )
    elements = np.char.upper(np.asarray(topology.element, dtype=str)[atom_idx])
    chain_ids = np.asarray(topology.chain_ids, dtype=str)[atom_idx]

    ff_vac = wk_vacuum_form_factors(elements, q_values)
    volumes = atomic_volumes_from_elements(elements)
    ff_excl = compute_ff_excl(atomic_volumes=volumes, q=q_values, rho0=cfg.rho0)
    water_curve = wk_water_form_factor(q_values, rho0=cfg.rho0)

    sasa_all = np.asarray(sasa(atom_array, vdw_radii="Single"), dtype=np.float32)
    solvent_acc = _solvent_accessibility_fraction(elements=elements, atom_sasa=sasa_all[atom_idx])
    ff_water = solvent_acc[:, None] * np.asarray(water_curve, dtype=np.float32)[None, :]

    return SAXSFeatures(
        topology=topology,
        output_index=output_index,
        atom_idx=atom_idx,
        ff_vac=ff_vac,
        ff_excl=ff_excl,
        ff_water=ff_water,
        solvent_acc=solvent_acc,
        q_values=q_values,
        chain_ids=chain_ids,
    )


__all__ = ["featurise"]
