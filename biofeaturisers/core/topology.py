import json
from dataclasses import dataclass, asdict
import numpy as np
from typing import Optional, Dict, Any

@dataclass
class MinimalTopology:
    atom_names: np.ndarray
    res_names: np.ndarray
    res_ids: np.ndarray
    chain_ids: np.ndarray
    elements: np.ndarray
    
    def to_dict(self) -> dict:
        return {
            "atom_names": self.atom_names.tolist(),
            "res_names": self.res_names.tolist(),
            "res_ids": self.res_ids.tolist(),
            "chain_ids": self.chain_ids.tolist(),
            "elements": self.elements.tolist(),
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
        )

    @classmethod
    def from_json(cls, filepath: str) -> "MinimalTopology":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # Temporary Biotite dummy integration
    @classmethod
    def from_biotite_dummy(cls, num_atoms: int = 10) -> "MinimalTopology":
        return cls(
            atom_names=np.array(["CA"] * num_atoms),
            res_names=np.array(["ALA"] * num_atoms),
            res_ids=np.arange(1, num_atoms + 1),
            chain_ids=np.array(["A"] * num_atoms),
            elements=np.array(["C"] * num_atoms)
        )
