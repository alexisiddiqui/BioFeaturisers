"""Tests for SAXS form-factor helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from biofeaturisers.saxs.form_factors import (
    atomic_volumes_from_elements,
    compute_ff_excl,
    compute_ff_table,
    compute_ff_water,
    wk_coefficients_for_elements,
    wk_vacuum_form_factors,
    wk_water_form_factor,
)


_ATOMIC_NUMBERS = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "P": 15.0, "S": 16.0}


def test_wk_q0_matches_atomic_number() -> None:
    q = jnp.asarray([0.0], dtype=jnp.float32)
    for element, atomic_number in _ATOMIC_NUMBERS.items():
        ff = wk_vacuum_form_factors([element], q)
        np.testing.assert_allclose(np.asarray(ff)[0, 0], atomic_number, atol=0.1, rtol=0.0)


def test_wk_positive_and_monotonic_decay() -> None:
    q = jnp.linspace(0.0, 0.5, 300, dtype=jnp.float32)
    for element in ("C", "N", "O", "S"):
        ff = np.asarray(wk_vacuum_form_factors([element], q))[0]
        assert np.all(ff > 0.0)
        assert np.all(np.diff(ff) <= 1e-4)


def test_wk_coeff_gather_consistency() -> None:
    q = jnp.linspace(0.01, 0.30, 50, dtype=jnp.float32)
    elements = np.asarray(["C", "N", "C", "O"], dtype=str)
    ff = np.asarray(wk_vacuum_form_factors(elements, q))
    np.testing.assert_allclose(ff[0], ff[2], atol=1e-6)
    assert not np.allclose(ff[0], ff[1], atol=1e-4)


def test_compute_ff_table_matches_vectorized_lookup() -> None:
    q = jnp.linspace(0.01, 0.5, 40, dtype=jnp.float32)
    symbols, _, a, b, c = wk_coefficients_for_elements(["C", "N", "O"])
    table = compute_ff_table(a=a, b=b, c=c, q=q)
    ff_lookup = wk_vacuum_form_factors(symbols, q)
    np.testing.assert_allclose(np.asarray(table), np.asarray(ff_lookup), atol=1e-6)


def test_excluded_volume_q0_is_rho0_times_volume() -> None:
    volumes = jnp.asarray([16.44], dtype=jnp.float32)
    rho0 = 0.334
    ff_excl = compute_ff_excl(volumes, q=jnp.asarray([0.0], dtype=jnp.float32), rho0=rho0)
    np.testing.assert_allclose(np.asarray(ff_excl)[0, 0], rho0 * 16.44, atol=1e-3, rtol=0.0)


def test_water_form_factor_formula() -> None:
    q = jnp.linspace(0.01, 0.2, 32, dtype=jnp.float32)
    ff_h = wk_vacuum_form_factors(["H"], q)[0]
    ff_o = wk_vacuum_form_factors(["O"], q)[0]
    water = compute_ff_water(ff_h=ff_h, ff_o=ff_o, q=q)
    auto = wk_water_form_factor(q)
    np.testing.assert_allclose(np.asarray(water), np.asarray(auto), atol=1e-6)


def test_atomic_volume_helper_is_positive() -> None:
    volumes = atomic_volumes_from_elements(["C", "N", "O", "S", "P"])
    assert volumes.shape == (5,)
    assert np.all(volumes > 0.0)

