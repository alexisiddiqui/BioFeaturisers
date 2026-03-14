"""SAXS feature dataclass contracts and persistence."""

from __future__ import annotations

from dataclasses import dataclass

import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from biofeaturisers.core.output_index import OutputIndex
from biofeaturisers.core.topology import MinimalTopology
from biofeaturisers.io.load import load_feature_bundle, output_index_from_arrays
from biofeaturisers.io.save import output_index_arrays, save_feature_bundle


def _as_str_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=str)


@dataclass(slots=True)
class SAXSFeatures:
    """Static SAXS featurisation contract."""

    topology: MinimalTopology
    output_index: OutputIndex
    atom_idx: Int[Array, "n_sel"]
    ff_vac: Float[Array, "n_sel n_q"]
    ff_excl: Float[Array, "n_sel n_q"]
    ff_water: Float[Array, "n_sel n_q"]
    solvent_acc: Float[Array, "n_sel"]
    q_values: Float[Array, "n_q"]
    chain_ids: np.ndarray

    def __post_init__(self) -> None:
        self.atom_idx = jnp.asarray(self.atom_idx, dtype=jnp.int32)
        self.ff_vac = jnp.asarray(self.ff_vac, dtype=jnp.float32)
        self.ff_excl = jnp.asarray(self.ff_excl, dtype=jnp.float32)
        self.ff_water = jnp.asarray(self.ff_water, dtype=jnp.float32)
        self.solvent_acc = jnp.asarray(self.solvent_acc, dtype=jnp.float32)
        self.q_values = jnp.asarray(self.q_values, dtype=jnp.float32)
        self.chain_ids = _as_str_array(self.chain_ids)

        chex.assert_rank(self.atom_idx, 1)
        chex.assert_rank(self.ff_vac, 2)
        chex.assert_rank(self.ff_excl, 2)
        chex.assert_rank(self.ff_water, 2)
        chex.assert_rank(self.solvent_acc, 1)
        chex.assert_rank(self.q_values, 1)

        n_atoms = int(self.atom_idx.shape[0])
        n_q = int(self.q_values.shape[0])
        if self.ff_vac.shape != (n_atoms, n_q):
            raise ValueError("ff_vac shape must be (n_selected_atoms, n_q)")
        if self.ff_excl.shape != (n_atoms, n_q):
            raise ValueError("ff_excl shape must be (n_selected_atoms, n_q)")
        if self.ff_water.shape != (n_atoms, n_q):
            raise ValueError("ff_water shape must be (n_selected_atoms, n_q)")
        if int(self.solvent_acc.shape[0]) != n_atoms:
            raise ValueError("solvent_acc length must match selected atom count")
        if int(self.chain_ids.shape[0]) != n_atoms:
            raise ValueError("chain_ids length must match selected atom count")

    def save(self, prefix: str) -> None:
        """Persist static SAXS features as NPZ + topology JSON."""
        save_feature_bundle(
            prefix=prefix,
            topology=self.topology,
            arrays={
                "atom_idx": np.asarray(self.atom_idx, dtype=np.int32),
                "ff_vac": np.asarray(self.ff_vac, dtype=np.float32),
                "ff_excl": np.asarray(self.ff_excl, dtype=np.float32),
                "ff_water": np.asarray(self.ff_water, dtype=np.float32),
                "solvent_acc": np.asarray(self.solvent_acc, dtype=np.float32),
                "q_values": np.asarray(self.q_values, dtype=np.float32),
                "chain_ids": self.chain_ids,
                **output_index_arrays(self.output_index),
            },
        )

    @classmethod
    def load(cls, prefix: str) -> "SAXSFeatures":
        """Load SAXS features from NPZ + topology JSON."""
        topology, data = load_feature_bundle(prefix)
        output_index = output_index_from_arrays(data)

        return cls(
            topology=topology,
            output_index=output_index,
            atom_idx=data["atom_idx"],
            ff_vac=data["ff_vac"],
            ff_excl=data["ff_excl"],
            ff_water=data["ff_water"],
            solvent_acc=data["solvent_acc"],
            q_values=data["q_values"],
            chain_ids=data["chain_ids"],
        )
