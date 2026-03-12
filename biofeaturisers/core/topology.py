import json
from dataclasses import dataclass, asdict
import numpy as np
from typing import Optional, Dict, Any

@dataclass
class MinimalTopology:
    # Per-atom arrays
    atom_names:    np.ndarray   # (N,) str
    res_names:     np.ndarray   # (N,) str
    res_ids:       np.ndarray   # (N,) int
    chain_ids:     np.ndarray   # (N,) str
    elements:      np.ndarray   # (N,) str
    is_hetatm:     np.ndarray   # (N,) bool
    is_backbone:   np.ndarray   # (N,) bool
    seg_ids:       np.ndarray   # (N,) str

    # Per-residue arrays
    res_unique_ids:   np.ndarray # (R,) str — "A:42" etc
    res_can_exchange: np.ndarray # (R,) bool

    def to_dict(self) -> dict:
        return {
            "atom_names": self.atom_names.tolist(),
            "res_names": self.res_names.tolist(),
            "res_ids": self.res_ids.tolist(),
            "chain_ids": self.chain_ids.tolist(),
            "elements": self.elements.tolist(),
            "is_hetatm": self.is_hetatm.tolist(),
            "is_backbone": self.is_backbone.tolist(),
            "seg_ids": self.seg_ids.tolist(),
            "res_unique_ids": self.res_unique_ids.tolist(),
            "res_can_exchange": self.res_can_exchange.tolist(),
        }

    def to_json(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, data: dict) -> "MinimalTopology":
        return cls(
            atom_names=np.array(data["atom_names"], dtype=str),
            res_names=np.array(data["res_names"], dtype=str),
            res_ids=np.array(data["res_ids"], dtype=int),
            chain_ids=np.array(data["chain_ids"], dtype=str),
            elements=np.array(data["elements"], dtype=str),
            is_hetatm=np.array(data["is_hetatm"], dtype=bool),
            is_backbone=np.array(data["is_backbone"], dtype=bool),
            seg_ids=np.array(data["seg_ids"], dtype=str),
            res_unique_ids=np.array(data["res_unique_ids"], dtype=str),
            res_can_exchange=np.array(data["res_can_exchange"], dtype=bool),
        )

    @classmethod
    def from_json(cls, filepath: str) -> "MinimalTopology":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # Temporary Biotite dummy integration
    @classmethod
    def from_biotite_dummy(cls, num_atoms: int = 10) -> "MinimalTopology":
        # Assume 1 atom per residue for dummy
        res_unique_ids = np.array([f"A:{i+1}" for i in range(num_atoms)])
        return cls(
            atom_names=np.array(["CA"] * num_atoms),
            res_names=np.array(["ALA"] * num_atoms),
            res_ids=np.arange(1, num_atoms + 1),
            chain_ids=np.array(["A"] * num_atoms),
            elements=np.array(["C"] * num_atoms),
            is_hetatm=np.zeros(num_atoms, dtype=bool),
            is_backbone=np.ones(num_atoms, dtype=bool),
            seg_ids=np.array([""] * num_atoms),
            res_unique_ids=res_unique_ids,
            res_can_exchange=np.ones(num_atoms, dtype=bool)
        )
