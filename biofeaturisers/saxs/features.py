"""SAXS feature dataclass contracts and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology


@dataclass(slots=True)
class SAXSFeatures:
    """Static SAXS featurisation contract."""

    topology: MinimalTopology
    output_index: OutputIndex
    atom_idx: np.ndarray
    ff_vac: np.ndarray
    ff_excl: np.ndarray
    ff_water: np.ndarray
    solvent_acc: np.ndarray
    q_values: np.ndarray
    chain_ids: np.ndarray

    def __post_init__(self) -> None:
        n_atoms = self.atom_idx.shape[0]
        n_q = self.q_values.shape[0]
        if self.ff_vac.shape != (n_atoms, n_q):
            raise ValueError("ff_vac shape must be (n_selected_atoms, n_q)")
        if self.ff_excl.shape != (n_atoms, n_q):
            raise ValueError("ff_excl shape must be (n_selected_atoms, n_q)")
        if self.ff_water.shape != (n_atoms, n_q):
            raise ValueError("ff_water shape must be (n_selected_atoms, n_q)")
        if self.solvent_acc.shape[0] != n_atoms:
            raise ValueError("solvent_acc length must match selected atom count")
        if self.chain_ids.shape[0] != n_atoms:
            raise ValueError("chain_ids length must match selected atom count")

    def save(self, prefix: str) -> None:
        """Persist static SAXS features as NPZ + topology JSON."""
        prefix_path = Path(prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        features_path = prefix_path.parent / f"{prefix_path.name}_features.npz"
        topology_path = prefix_path.parent / f"{prefix_path.name}_topology.json"

        np.savez(
            features_path,
            atom_idx=self.atom_idx,
            ff_vac=self.ff_vac,
            ff_excl=self.ff_excl,
            ff_water=self.ff_water,
            solvent_acc=self.solvent_acc,
            q_values=self.q_values,
            chain_ids=self.chain_ids,
            output_atom_mask=self.output_index.atom_mask,
            output_probe_mask=self.output_index.probe_mask,
            output_mask=self.output_index.output_mask,
            output_atom_idx=self.output_index.atom_idx,
            output_probe_idx=self.output_index.probe_idx,
            output_res_idx=self.output_index.output_res_idx,
        )

        with topology_path.open("w", encoding="utf-8") as handle:
            json.dump(self.topology.to_json(), handle)

    @classmethod
    def load(cls, prefix: str) -> "SAXSFeatures":
        """Load SAXS features from NPZ + topology JSON."""
        prefix_path = Path(prefix)
        features_path = prefix_path.parent / f"{prefix_path.name}_features.npz"
        topology_path = prefix_path.parent / f"{prefix_path.name}_topology.json"

        with topology_path.open("r", encoding="utf-8") as handle:
            topology = MinimalTopology.from_json(json.load(handle))

        data = np.load(features_path, allow_pickle=False)
        output_index = OutputIndex(
            atom_mask=data["output_atom_mask"].astype(bool),
            probe_mask=data["output_probe_mask"].astype(bool),
            output_mask=data["output_mask"].astype(bool),
            atom_idx=data["output_atom_idx"].astype(np.int32),
            probe_idx=data["output_probe_idx"].astype(np.int32),
            output_res_idx=data["output_res_idx"].astype(np.int32),
        )

        return cls(
            topology=topology,
            output_index=output_index,
            atom_idx=data["atom_idx"].astype(np.int32),
            ff_vac=data["ff_vac"].astype(np.float32),
            ff_excl=data["ff_excl"].astype(np.float32),
            ff_water=data["ff_water"].astype(np.float32),
            solvent_acc=data["solvent_acc"].astype(np.float32),
            q_values=data["q_values"].astype(np.float32),
            chain_ids=data["chain_ids"].astype(str),
        )

