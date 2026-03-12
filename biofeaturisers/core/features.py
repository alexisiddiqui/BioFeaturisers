from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .topology import MinimalTopology

@dataclass
class HDXFeatures:
    """
    Features required for HDX forward computation.
    """
    topology: MinimalTopology

    # Probe atom indices
    amide_N_idx: np.ndarray         # (N_res,) int32
    amide_H_idx: np.ndarray         # (N_res,) int32 (observed H index when present)
    amide_CA_idx: np.ndarray        # (N_res,) int32
    amide_C_prev_idx: np.ndarray    # (N_res,) int32 (C atom of previous residue)
    amide_has_observed_H: np.ndarray # (N_res,) bool

    # Environment atom indices
    heavy_atom_idx: np.ndarray      # (N_heavy,) int32
    backbone_O_idx: np.ndarray      # (N_bb_O,) int32

    # Exclusion masks
    excl_mask_c: np.ndarray         # (N_res, N_heavy) float32
    excl_mask_h: np.ndarray         # (N_res, N_bb_O) float32

    # Residue-aligned metadata
    res_keys: np.ndarray            # (N_res,) str
    res_names: np.ndarray           # (N_res,) str
    can_exchange: np.ndarray        # (N_res,) bool
    kint: np.ndarray | None = None  # (N_res,) float32 or None

    def save(self, prefix: str) -> None:
        payload: dict[str, np.ndarray] = {
            "amide_N_idx": self.amide_N_idx.astype(np.int32),
            "amide_H_idx": self.amide_H_idx.astype(np.int32),
            "amide_CA_idx": self.amide_CA_idx.astype(np.int32),
            "amide_C_prev_idx": self.amide_C_prev_idx.astype(np.int32),
            "amide_has_observed_H": self.amide_has_observed_H.astype(bool),
            "heavy_atom_idx": self.heavy_atom_idx.astype(np.int32),
            "backbone_O_idx": self.backbone_O_idx.astype(np.int32),
            "excl_mask_c": self.excl_mask_c.astype(np.float32),
            "excl_mask_h": self.excl_mask_h.astype(np.float32),
            "res_keys": self.res_keys.astype(str),
            "res_names": self.res_names.astype(str),
            "can_exchange": self.can_exchange.astype(bool),
            "has_kint": np.array([self.kint is not None], dtype=bool),
        }
        if self.kint is not None:
            payload["kint"] = self.kint.astype(np.float32)

        np.savez(f"{prefix}_features.npz", **payload)
        self.topology.to_json(f"{prefix}_topology.json")

    @classmethod
    def load(cls, prefix: str) -> "HDXFeatures":
        with np.load(f"{prefix}_features.npz", allow_pickle=False) as data:
            has_kint = bool(data["has_kint"][0])
            kint = data["kint"].astype(np.float32) if has_kint and "kint" in data.files else None
            return cls(
                topology=MinimalTopology.from_json(f"{prefix}_topology.json"),
                amide_N_idx=data["amide_N_idx"].astype(np.int32),
                amide_H_idx=data["amide_H_idx"].astype(np.int32),
                amide_CA_idx=data["amide_CA_idx"].astype(np.int32),
                amide_C_prev_idx=data["amide_C_prev_idx"].astype(np.int32),
                amide_has_observed_H=data["amide_has_observed_H"].astype(bool),
                heavy_atom_idx=data["heavy_atom_idx"].astype(np.int32),
                backbone_O_idx=data["backbone_O_idx"].astype(np.int32),
                excl_mask_c=data["excl_mask_c"].astype(np.float32),
                excl_mask_h=data["excl_mask_h"].astype(np.float32),
                res_keys=data["res_keys"].astype(str),
                res_names=data["res_names"].astype(str),
                can_exchange=data["can_exchange"].astype(bool),
                kint=kint,
            )

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
