"""Minimal serialisable topology contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


def _as_str_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=str)


@dataclass(slots=True)
class MinimalTopology:
    """Minimal atom- and residue-level topology arrays used by featurisers."""

    atom_names: np.ndarray
    res_names: np.ndarray
    res_ids: np.ndarray
    chain_ids: np.ndarray
    element: np.ndarray
    is_hetatm: np.ndarray
    is_backbone: np.ndarray
    seg_ids: np.ndarray
    res_unique_ids: np.ndarray
    res_can_exchange: np.ndarray

    def __post_init__(self) -> None:
        n_atoms = self.atom_names.shape[0]
        atom_fields = (
            ("res_names", self.res_names),
            ("res_ids", self.res_ids),
            ("chain_ids", self.chain_ids),
            ("element", self.element),
            ("is_hetatm", self.is_hetatm),
            ("is_backbone", self.is_backbone),
            ("seg_ids", self.seg_ids),
        )
        for field_name, field_value in atom_fields:
            if field_value.shape[0] != n_atoms:
                raise ValueError(
                    f"{field_name} length {field_value.shape[0]} != atom_names length {n_atoms}"
                )
        if self.res_can_exchange.shape[0] != self.res_unique_ids.shape[0]:
            raise ValueError("res_can_exchange length must match res_unique_ids length")

    def to_json(self) -> dict:
        """Convert topology into JSON-safe primitive collections."""
        return {
            "atom_names": self.atom_names.tolist(),
            "res_names": self.res_names.tolist(),
            "res_ids": self.res_ids.tolist(),
            "chain_ids": self.chain_ids.tolist(),
            "element": self.element.tolist(),
            "is_hetatm": self.is_hetatm.tolist(),
            "is_backbone": self.is_backbone.tolist(),
            "seg_ids": self.seg_ids.tolist(),
            "res_unique_ids": self.res_unique_ids.tolist(),
            "res_can_exchange": self.res_can_exchange.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict) -> "MinimalTopology":
        """Reconstruct topology from JSON payload."""
        return cls(
            atom_names=_as_str_array(data["atom_names"]),
            res_names=_as_str_array(data["res_names"]),
            res_ids=np.asarray(data["res_ids"], dtype=np.int32),
            chain_ids=_as_str_array(data["chain_ids"]),
            element=_as_str_array(data["element"]),
            is_hetatm=np.asarray(data["is_hetatm"], dtype=bool),
            is_backbone=np.asarray(data["is_backbone"], dtype=bool),
            seg_ids=_as_str_array(data["seg_ids"]),
            res_unique_ids=_as_str_array(data["res_unique_ids"]),
            res_can_exchange=np.asarray(data["res_can_exchange"], dtype=bool),
        )

    @classmethod
    def from_biotite(
        cls, atom_array: object, exchange_mask: Iterable[str] | None = None
    ) -> "MinimalTopology":
        """Create topology from a biotite AtomArray-like object."""

        atom_names = _as_str_array(_require_attr(atom_array, ("atom_name",)))
        n_atoms = atom_names.shape[0]
        res_names = _as_str_array(_require_attr(atom_array, ("res_name",)))
        res_ids = np.asarray(_require_attr(atom_array, ("res_id",)), dtype=np.int32)
        chain_ids = _as_str_array(_require_attr(atom_array, ("chain_id",)))
        element = _as_str_array(_require_attr(atom_array, ("element",)))
        is_hetatm = np.asarray(
            _require_attr(atom_array, ("hetero", "is_hetero", "is_hetatm"), np.zeros(n_atoms)),
            dtype=bool,
        )
        seg_ids = _as_str_array(_require_attr(atom_array, ("seg_id",), np.full(n_atoms, "")))
        is_backbone = np.isin(atom_names, tuple(_BACKBONE_ATOMS))

        residue_pairs = sorted(
            {(str(chain), int(res_id)) for chain, res_id in zip(chain_ids, res_ids, strict=True)}
        )
        res_unique_ids = _as_str_array([f"{chain}:{res_id}" for chain, res_id in residue_pairs])
        res_can_exchange = np.ones(len(residue_pairs), dtype=bool)

        first_pair_per_chain: dict[str, tuple[str, int]] = {}
        res_name_by_pair: dict[tuple[str, int], str] = {}
        has_hetatm_by_pair: dict[tuple[str, int], bool] = {}

        for chain, res_id, res_name, het in zip(
            chain_ids, res_ids, res_names, is_hetatm, strict=True
        ):
            pair = (str(chain), int(res_id))
            first_pair_per_chain.setdefault(str(chain), pair)
            res_name_by_pair.setdefault(pair, str(res_name))
            has_hetatm_by_pair[pair] = has_hetatm_by_pair.get(pair, False) or bool(het)

        for idx, pair in enumerate(residue_pairs):
            if pair == first_pair_per_chain[pair[0]]:
                res_can_exchange[idx] = False
            if res_name_by_pair.get(pair, "").upper() == "PRO":
                res_can_exchange[idx] = False
            if has_hetatm_by_pair.get(pair, False):
                res_can_exchange[idx] = False

        if exchange_mask:
            blocked = {str(value) for value in exchange_mask}
            for idx, key in enumerate(res_unique_ids):
                if str(key) in blocked:
                    res_can_exchange[idx] = False

        return cls(
            atom_names=atom_names,
            res_names=res_names,
            res_ids=res_ids,
            chain_ids=chain_ids,
            element=element,
            is_hetatm=is_hetatm,
            is_backbone=is_backbone,
            seg_ids=seg_ids,
            res_unique_ids=res_unique_ids,
            res_can_exchange=res_can_exchange,
        )


def _require_attr(obj: object, names: tuple[str, ...], default: object | None = None) -> object:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not None:
        return default
    joined = ", ".join(names)
    raise AttributeError(f"Missing required attribute; expected one of: {joined}")

