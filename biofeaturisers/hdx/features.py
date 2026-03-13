"""HDX feature dataclass contracts and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology


@dataclass(slots=True)
class HDXFeatures:
    """Static HDX featurisation contract."""

    topology: MinimalTopology
    output_index: OutputIndex
    amide_N_idx: np.ndarray
    amide_H_idx: np.ndarray
    heavy_atom_idx: np.ndarray
    backbone_O_idx: np.ndarray
    excl_mask_c: np.ndarray
    excl_mask_h: np.ndarray
    res_keys: np.ndarray
    res_names: np.ndarray
    can_exchange: np.ndarray
    kint: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_probe = self.amide_N_idx.shape[0]
        if self.amide_H_idx.shape[0] != n_probe:
            raise ValueError("amide_N_idx and amide_H_idx must have the same length")
        if self.excl_mask_c.shape[0] != n_probe:
            raise ValueError("excl_mask_c row count must match probe count")
        if self.excl_mask_h.shape[0] != n_probe:
            raise ValueError("excl_mask_h row count must match probe count")
        n_out = self.res_keys.shape[0]
        if self.res_names.shape[0] != n_out or self.can_exchange.shape[0] != n_out:
            raise ValueError("res_keys/res_names/can_exchange lengths must match")
        if self.output_index.output_res_idx.shape[0] != n_out:
            raise ValueError("output_res_idx length must match HDX output residue metadata length")
        if self.kint is not None and self.kint.shape[0] != n_out:
            raise ValueError("kint length must match output residue length")

    def save(self, prefix: str) -> None:
        """Persist static HDX features as NPZ + topology JSON."""
        prefix_path = Path(prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        features_path = prefix_path.parent / f"{prefix_path.name}_features.npz"
        topology_path = prefix_path.parent / f"{prefix_path.name}_topology.json"

        np.savez(
            features_path,
            amide_N_idx=self.amide_N_idx,
            amide_H_idx=self.amide_H_idx,
            heavy_atom_idx=self.heavy_atom_idx,
            backbone_O_idx=self.backbone_O_idx,
            excl_mask_c=self.excl_mask_c,
            excl_mask_h=self.excl_mask_h,
            res_keys=self.res_keys,
            res_names=self.res_names,
            can_exchange=self.can_exchange,
            output_atom_mask=self.output_index.atom_mask,
            output_probe_mask=self.output_index.probe_mask,
            output_mask=self.output_index.output_mask,
            output_atom_idx=self.output_index.atom_idx,
            output_probe_idx=self.output_index.probe_idx,
            output_res_idx=self.output_index.output_res_idx,
            has_kint=np.asarray(self.kint is not None, dtype=bool),
            kint=self.kint if self.kint is not None else np.asarray([], dtype=np.float32),
        )

        with topology_path.open("w", encoding="utf-8") as handle:
            json.dump(self.topology.to_json(), handle)

    @classmethod
    def load(cls, prefix: str) -> "HDXFeatures":
        """Load HDX features from NPZ + topology JSON."""
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
        kint = data["kint"].astype(np.float32) if bool(data["has_kint"]) else None

        return cls(
            topology=topology,
            output_index=output_index,
            amide_N_idx=data["amide_N_idx"].astype(np.int32),
            amide_H_idx=data["amide_H_idx"].astype(np.int32),
            heavy_atom_idx=data["heavy_atom_idx"].astype(np.int32),
            backbone_O_idx=data["backbone_O_idx"].astype(np.int32),
            excl_mask_c=data["excl_mask_c"].astype(np.float32),
            excl_mask_h=data["excl_mask_h"].astype(np.float32),
            res_keys=data["res_keys"].astype(str),
            res_names=data["res_names"].astype(str),
            can_exchange=data["can_exchange"].astype(bool),
            kint=kint,
        )

