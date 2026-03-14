"""Tests for SAXS hydration parameter fitting."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from biofeaturisers.config import SAXSConfig
from biofeaturisers.saxs.foxs import saxs_combine
from biofeaturisers.saxs.hydration import fit_c1_c2, fit_c1_c2_analytic, scaled_reduced_chi2


def test_fit_c1_c2_recovers_synthetic_minimum() -> None:
    q = 12
    partials = jnp.asarray(
        [
            np.linspace(10.0, 8.0, q),
            np.linspace(1.0, 0.4, q),
            np.linspace(0.3, 0.1, q),
            np.linspace(2.0, 1.0, q),
            np.linspace(1.2, 0.6, q),
            np.linspace(0.7, 0.2, q),
        ],
        dtype=jnp.float32,
    )
    c1_true, c2_true = 1.05, 2.0
    i_calc = saxs_combine(partials, c1=c1_true, c2=c2_true)
    i_exp = 2.7 * i_calc
    sigma = jnp.ones_like(i_exp) * 0.05

    config = SAXSConfig(
        c1_range=(1.0, 1.1),
        c2_range=(1.0, 3.0),
        c1_steps=11,
        c2_steps=21,
    )
    c1_fit, c2_fit, chi2 = fit_c1_c2(partials=partials, i_exp=i_exp, sigma=sigma, config=config)
    np.testing.assert_allclose(c1_fit, c1_true, atol=1e-6)
    np.testing.assert_allclose(c2_fit, c2_true, atol=1e-6)
    assert chi2 < 1e-6


def test_scaled_reduced_chi2_and_alias_agree() -> None:
    partials = jnp.asarray(
        [
            [9.0, 8.5, 8.1],
            [0.8, 0.7, 0.6],
            [0.2, 0.2, 0.2],
            [1.3, 1.2, 1.1],
            [0.7, 0.6, 0.5],
            [0.4, 0.3, 0.2],
        ],
        dtype=jnp.float32,
    )
    i_exp = saxs_combine(partials, c1=1.03, c2=1.5)
    sigma = jnp.asarray([0.1, 0.1, 0.1], dtype=jnp.float32)
    chi2 = scaled_reduced_chi2(i_calc=i_exp, i_exp=i_exp, sigma=sigma)
    np.testing.assert_allclose(np.asarray(chi2), np.asarray(0.0, dtype=np.float32), atol=1e-7)

    config = SAXSConfig(c1_range=(1.0, 1.06), c2_range=(1.0, 1.6), c1_steps=7, c2_steps=7)
    direct = fit_c1_c2(partials=partials, i_exp=i_exp, sigma=sigma, config=config)
    alias = fit_c1_c2_analytic(partials=partials, i_exp=i_exp, sigma=sigma, config=config)
    np.testing.assert_allclose(np.asarray(alias), np.asarray(direct), atol=1e-8)

