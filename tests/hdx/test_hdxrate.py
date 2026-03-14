"""Tests for HDXrate integration utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.core.topology import MinimalTopology
from biofeaturisers.hdx import hdxrate


def _make_two_chain_topology() -> MinimalTopology:
    return MinimalTopology(
        atom_names=np.asarray(["N", "N", "N", "N", "N", "N"], dtype=str),
        res_names=np.asarray(["ALA", "GLY", "SER", "ALA", "GLY", "SER"], dtype=str),
        res_ids=np.asarray([1, 2, 3, 1, 2, 3], dtype=np.int32),
        chain_ids=np.asarray(["A", "A", "A", "B", "B", "B"], dtype=str),
        element=np.asarray(["N", "N", "N", "N", "N", "N"], dtype=str),
        is_hetatm=np.asarray([False, False, False, False, False, False], dtype=bool),
        is_backbone=np.asarray([True, True, True, True, True, True], dtype=bool),
        seg_ids=np.asarray(["", "", "", "", "", ""], dtype=str),
        res_unique_ids=np.asarray(["A:1", "A:2", "A:3", "B:1", "B:2", "B:3"], dtype=str),
        res_can_exchange=np.asarray([False, True, True, False, True, True], dtype=bool),
    )


def _make_concatenated_topology() -> MinimalTopology:
    return MinimalTopology(
        atom_names=np.asarray(["N", "N", "N", "N", "N", "N"], dtype=str),
        res_names=np.asarray(["ALA", "GLY", "SER", "ALA", "GLY", "SER"], dtype=str),
        res_ids=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32),
        chain_ids=np.asarray(["A", "A", "A", "A", "A", "A"], dtype=str),
        element=np.asarray(["N", "N", "N", "N", "N", "N"], dtype=str),
        is_hetatm=np.asarray([False, False, False, False, False, False], dtype=bool),
        is_backbone=np.asarray([True, True, True, True, True, True], dtype=bool),
        seg_ids=np.asarray(["", "", "", "", "", ""], dtype=str),
        res_unique_ids=np.asarray(["A:1", "A:2", "A:3", "A:4", "A:5", "A:6"], dtype=str),
        res_can_exchange=np.asarray([False, True, True, True, True, True], dtype=bool),
    )


def _make_proline_topology() -> MinimalTopology:
    return MinimalTopology(
        atom_names=np.asarray(["N", "N", "N"], dtype=str),
        res_names=np.asarray(["ALA", "PRO", "GLY"], dtype=str),
        res_ids=np.asarray([1, 2, 3], dtype=np.int32),
        chain_ids=np.asarray(["A", "A", "A"], dtype=str),
        element=np.asarray(["N", "N", "N"], dtype=str),
        is_hetatm=np.asarray([False, False, False], dtype=bool),
        is_backbone=np.asarray([True, True, True], dtype=bool),
        seg_ids=np.asarray(["", "", ""], dtype=str),
        res_unique_ids=np.asarray(["A:1", "A:2", "A:3"], dtype=str),
        res_can_exchange=np.asarray([False, False, True], dtype=bool),
    )


def test_compute_kint_calls_hdxrate_once_per_chain(monkeypatch) -> None:
    topology = _make_two_chain_topology()
    calls: list[str] = []

    def fake_k_int_from_sequence(sequence: str, temperature: float, pH: float) -> np.ndarray:
        calls.append(sequence)
        assert temperature == 298.15
        assert pH == 7.0
        return np.linspace(0.0, 5.0, num=len(sequence), dtype=np.float32)

    monkeypatch.setattr(hdxrate, "_load_hdxrate_api", lambda: fake_k_int_from_sequence)
    kint = hdxrate.compute_kint(topology=topology, pH=7.0, temperature=298.15)

    assert calls == ["AGS", "AGS"]
    assert np.isnan(kint[0])
    assert np.isnan(kint[3])
    assert np.isfinite(kint[1])
    assert np.isfinite(kint[2])
    assert np.isfinite(kint[4])
    assert np.isfinite(kint[5])


def test_compute_kint_keeps_non_exchangeable_proline_as_nan(monkeypatch) -> None:
    topology = _make_proline_topology()

    def fake_k_int_from_sequence(sequence: str, temperature: float, pH: float) -> np.ndarray:
        assert sequence == "APG"
        assert temperature == 298.15
        assert pH == 7.0
        return np.asarray([0.0, 2.5, 3.5], dtype=np.float32)

    monkeypatch.setattr(hdxrate, "_load_hdxrate_api", lambda: fake_k_int_from_sequence)
    kint = hdxrate.compute_kint(topology=topology, pH=7.0, temperature=298.15)

    assert np.isnan(kint[0])  # chain N-terminus
    assert np.isnan(kint[1])  # proline (non-exchangeable in topology mask)
    assert np.isfinite(kint[2]) and kint[2] > 0.0


def test_compute_kint_chain_concatenation_guard(monkeypatch) -> None:
    multi_chain = _make_two_chain_topology()
    concatenated = _make_concatenated_topology()

    def fake_k_int_from_sequence(sequence: str, temperature: float, pH: float) -> np.ndarray:
        base = np.arange(len(sequence), dtype=np.float32)
        base[0] = 0.0
        return base

    monkeypatch.setattr(hdxrate, "_load_hdxrate_api", lambda: fake_k_int_from_sequence)
    kint_multi = hdxrate.compute_kint(topology=multi_chain, pH=7.0, temperature=298.15)
    kint_concat = hdxrate.compute_kint(topology=concatenated, pH=7.0, temperature=298.15)

    assert np.isnan(kint_multi[3])  # B:1 should be N-terminus in per-chain evaluation
    assert np.isfinite(kint_concat[3]) and kint_concat[3] > 0.0  # A:4 is not chain N-term


def test_predict_uptake_matches_expected_limits() -> None:
    ln_pf = jnp.asarray([10.0, 0.0], dtype=jnp.float32)
    kint = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    can_exchange = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    peptide_mask = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    uptake = hdxrate.predict_uptake(
        ln_pf=ln_pf,
        kint=kint,
        can_exchange=can_exchange,
        peptide_mask=peptide_mask,
        timepoints=(1.0,),
    )

    expected_slow = 1.0 - np.exp(-np.exp(-10.0))
    expected_fast = 1.0 - np.exp(-1.0)
    np.testing.assert_allclose(float(uptake[0, 0]), expected_slow, rtol=1e-3, atol=1e-8)
    np.testing.assert_allclose(float(uptake[1, 0]), expected_fast, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(float(uptake[2, 0]), expected_slow + expected_fast, rtol=1e-5)


def test_predict_uptake_gradient_is_finite_and_non_positive_wrt_ln_pf() -> None:
    kint = jnp.asarray([1.0, 2.0, 0.5], dtype=jnp.float32)
    can_exchange = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)
    peptide_mask = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=jnp.float32)
    timepoints = (0.5, 5.0)

    def loss(ln_pf: jax.Array) -> jax.Array:
        uptake = hdxrate.predict_uptake(
            ln_pf=ln_pf,
            kint=kint,
            can_exchange=can_exchange,
            peptide_mask=peptide_mask,
            timepoints=timepoints,
        )
        return jnp.sum(uptake)

    ln_pf = jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float32)
    grad = jax.grad(loss)(ln_pf)
    grad_np = np.asarray(grad)
    assert np.isfinite(grad_np).all()
    assert np.all(grad_np <= 1e-6)
