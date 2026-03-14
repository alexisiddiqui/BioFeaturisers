"""Gradient tests for the public SAXS ``forward`` wrapper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from biofeaturisers.config import SAXSConfig
from biofeaturisers.saxs.features import SAXSFeatures
from biofeaturisers.saxs.forward import forward
from tests.fixtures.numerical_helpers import assert_directional_gradient_close


def _make_features(simple_topology, simple_output_index, ff_vac, ff_excl, ff_water, q_values):
    n_sel = int(ff_vac.shape[0])
    return SAXSFeatures(
        topology=simple_topology,
        output_index=simple_output_index,
        atom_idx=jnp.arange(n_sel, dtype=jnp.int32),
        ff_vac=jnp.asarray(ff_vac, dtype=jnp.float32),
        ff_excl=jnp.asarray(ff_excl, dtype=jnp.float32),
        ff_water=jnp.asarray(ff_water, dtype=jnp.float32),
        solvent_acc=jnp.ones((n_sel,), dtype=jnp.float32),
        q_values=jnp.asarray(q_values, dtype=jnp.float32),
        chain_ids=np.asarray(["A"] * n_sel, dtype=str),
    )


def test_forward_single_structure_gradient_matches_fd(
    simple_topology,
    simple_output_index,
) -> None:
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.2, -0.1],
            [2.6, 0.9, 0.3],
        ],
        dtype=jnp.float32,
    )
    q_values = jnp.linspace(0.02, 0.30, 7, dtype=jnp.float32)
    ff_vac = jnp.asarray(
        [
            [1.00, 0.94, 0.88, 0.81, 0.75, 0.70, 0.66],
            [0.93, 0.87, 0.82, 0.77, 0.72, 0.67, 0.63],
            [0.86, 0.81, 0.77, 0.72, 0.68, 0.64, 0.60],
        ],
        dtype=jnp.float32,
    )
    ff_excl = ff_vac * 0.30
    ff_water = ff_vac * 0.12
    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=ff_excl,
        ff_water=ff_water,
        q_values=q_values,
    )
    config = SAXSConfig(n_q=int(q_values.shape[0]), chunk_size=2, batch_size=2, fit_c1_c2=False)

    loss = lambda c: jnp.sum(forward(coords=c, features=features, config=config, c1=1.03, c2=0.4))
    assert_directional_gradient_close(loss, coords, eps=2e-3, n_dirs=3, seed=17, rtol=2e-1, atol=2e-2)


def test_forward_trajectory_gradient_weighted_path_is_finite(
    simple_topology,
    simple_output_index,
) -> None:
    base = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.3, -0.2],
            [2.4, 0.8, 0.4],
        ],
        dtype=jnp.float32,
    )
    trajectory = jnp.stack(
        [
            base,
            base + jnp.asarray([2.0, -0.7, 1.1], dtype=jnp.float32),
        ],
        axis=0,
    )
    weights = jnp.asarray([0.65, 0.35], dtype=jnp.float32)

    q_values = jnp.linspace(0.02, 0.28, 6, dtype=jnp.float32)
    ff_vac = jnp.asarray(
        [
            [1.00, 0.95, 0.90, 0.84, 0.78, 0.72],
            [0.92, 0.88, 0.83, 0.79, 0.74, 0.69],
            [0.85, 0.80, 0.76, 0.71, 0.67, 0.63],
        ],
        dtype=jnp.float32,
    )
    ff_excl = ff_vac * 0.25
    ff_water = ff_vac * 0.08
    features = _make_features(
        simple_topology=simple_topology,
        simple_output_index=simple_output_index,
        ff_vac=ff_vac,
        ff_excl=ff_excl,
        ff_water=ff_water,
        q_values=q_values,
    )
    config = SAXSConfig(n_q=int(q_values.shape[0]), chunk_size=2, batch_size=2, fit_c1_c2=False)

    loss = lambda traj: jnp.sum(
        forward(coords=traj, features=features, config=config, c1=1.01, c2=0.25, weights=weights)
    )

    grad = jax.grad(loss)(trajectory)
    assert grad.shape == trajectory.shape
    assert np.isfinite(np.asarray(grad)).all()

    assert_directional_gradient_close(
        loss,
        trajectory,
        eps=2e-3,
        n_dirs=2,
        seed=23,
        rtol=3e-1,
        atol=3e-2,
    )

