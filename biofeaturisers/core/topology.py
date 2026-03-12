from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

_BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}

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

    @property
    def atom_res_keys(self) -> np.ndarray:
        return np.array(
            [f"{chain}:{int(resid)}" for chain, resid in zip(self.chain_ids, self.res_ids, strict=True)],
            dtype=str,
        )

    def residue_name_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for key, res_name in zip(self.atom_res_keys, self.res_names, strict=True):
            if key not in mapping:
                mapping[key] = str(res_name)
        return mapping

    @classmethod
    def from_biotite(cls, atom_array: Any) -> "MinimalTopology":
        atom_names = np.asarray(atom_array.atom_name, dtype=str)
        res_names = np.asarray(atom_array.res_name, dtype=str)
        res_ids = np.asarray(atom_array.res_id, dtype=int)
        n_atoms = atom_names.shape[0]

        if hasattr(atom_array, "chain_id"):
            chain_ids = np.asarray(atom_array.chain_id, dtype=str)
        else:
            chain_ids = np.full(n_atoms, "", dtype=str)

        if hasattr(atom_array, "element"):
            elements = np.asarray(atom_array.element, dtype=str)
        else:
            # Best-effort fallback when element annotations are absent.
            elements = np.char.upper(np.char.strip(np.char.array([name[:1] for name in atom_names])))

        if hasattr(atom_array, "hetero"):
            is_hetatm = np.asarray(atom_array.hetero, dtype=bool)
        else:
            is_hetatm = np.zeros(n_atoms, dtype=bool)

        if hasattr(atom_array, "seg_id"):
            seg_ids = np.asarray(atom_array.seg_id, dtype=str)
        else:
            seg_ids = np.full(n_atoms, "", dtype=str)

        is_backbone = np.isin(atom_names, tuple(_BACKBONE_ATOM_NAMES))

        res_unique_ids: list[str] = []
        res_can_exchange: list[bool] = []
        seen_keys: set[str] = set()
        seen_chain: set[str] = set()
        for chain_id, res_id, res_name in zip(chain_ids, res_ids, res_names, strict=True):
            key = f"{chain_id}:{int(res_id)}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            res_unique_ids.append(key)

            is_n_term = str(chain_id) not in seen_chain
            if is_n_term:
                seen_chain.add(str(chain_id))
            res_can_exchange.append((str(res_name).upper() != "PRO") and (not is_n_term))

        return cls(
            atom_names=atom_names,
            res_names=res_names,
            res_ids=res_ids,
            chain_ids=chain_ids,
            elements=elements,
            is_hetatm=is_hetatm,
            is_backbone=is_backbone,
            seg_ids=seg_ids,
            res_unique_ids=np.asarray(res_unique_ids, dtype=str),
            res_can_exchange=np.asarray(res_can_exchange, dtype=bool),
        )

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
