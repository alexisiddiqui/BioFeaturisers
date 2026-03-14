"""FoXS-style SAXS form-factor helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from biotite.structure.info import vdw_radius_single
from jaxtyping import Array, Float

_WATER_VOLUME = 29.9


@dataclass(frozen=True, slots=True)
class _WKEntry:
    a: tuple[float, float, float, float, float]
    b: tuple[float, float, float, float, float]
    c: float


# Waasmaier-Kirfel (1995) parameters for common biomolecular elements.
# f0(k) = c + sum_i a_i exp(-b_i * k^2), where k = q / (4*pi)
_WK_COEFFS: dict[str, _WKEntry] = {
    "H": _WKEntry(
        a=(0.413048, 0.294953, 0.187491, 0.080701, 0.023736),
        b=(15.569946, 32.398468, 5.711404, 61.889874, 1.334118),
        c=0.000049,
    ),
    "C": _WKEntry(
        a=(2.657506, 1.078079, 1.490909, -4.241070, 0.713791),
        b=(14.780758, 0.776775, 42.086842, -0.000294, 0.239535),
        c=4.297983,
    ),
    "N": _WKEntry(
        a=(11.893780, 3.277479, 1.858092, 0.858927, 0.912985),
        b=(0.000158, 10.232723, 30.344690, 0.656065, 0.217287),
        c=-11.804902,
    ),
    "O": _WKEntry(
        a=(2.960427, 2.508818, 0.637853, 0.722838, 1.142756),
        b=(14.182259, 5.936858, 0.112726, 34.958481, 0.390240),
        c=0.027014,
    ),
    "S": _WKEntry(
        a=(6.372157, 5.154568, 1.473732, 1.635073, 1.209372),
        b=(1.514347, 22.092527, 0.061373, 55.445175, 0.646925),
        c=0.154722,
    ),
    "P": _WKEntry(
        a=(1.950541, 4.146930, 1.494560, 1.522042, 5.729711),
        b=(0.908139, 27.044952, 0.071280, 67.520187, 1.981173),
        c=0.155233,
    ),
}


def _normalise_element_symbol(symbol: str) -> str:
    clean = str(symbol).strip()
    if len(clean) == 0:
        raise ValueError("Element symbol cannot be empty")
    norm = clean[0].upper() + clean[1:].lower()
    if norm not in _WK_COEFFS:
        raise ValueError(f"Unsupported element for SAXS form factors: {symbol}")
    return norm


def _normalise_elements(elements: np.ndarray | list[str]) -> np.ndarray:
    return np.asarray([_normalise_element_symbol(symbol) for symbol in elements], dtype=str)


def compute_ff_table(
    a: Float[Array, "n_type 5"],
    b: Float[Array, "n_type 5"],
    c: Float[Array, "n_type"],
    q: Float[Array, "n_q"],
) -> Float[Array, "n_type n_q"]:
    """Evaluate Waasmaier-Kirfel 5-Gaussian form-factor tables."""
    a_arr = jnp.asarray(a, dtype=jnp.float32)
    b_arr = jnp.asarray(b, dtype=jnp.float32)
    c_arr = jnp.asarray(c, dtype=jnp.float32)
    q_arr = jnp.asarray(q, dtype=jnp.float32)

    s2 = (q_arr / (4.0 * jnp.pi)) ** 2
    exponents = -b_arr[:, :, None] * s2[None, None, :]
    return jnp.sum(a_arr[:, :, None] * jnp.exp(exponents), axis=1) + c_arr[:, None]


def wk_coefficients_for_elements(
    elements: np.ndarray | list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized unique symbols and WK coefficients."""
    symbols = _normalise_elements(elements)
    if symbols.shape[0] == 0:
        return (
            np.asarray([], dtype=str),
            np.asarray([], dtype=np.int32),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    unique_symbols, inverse = np.unique(symbols, return_inverse=True)

    a = np.stack([np.asarray(_WK_COEFFS[symbol].a, dtype=np.float32) for symbol in unique_symbols])
    b = np.stack([np.asarray(_WK_COEFFS[symbol].b, dtype=np.float32) for symbol in unique_symbols])
    c = np.asarray([_WK_COEFFS[symbol].c for symbol in unique_symbols], dtype=np.float32)
    return unique_symbols, inverse.astype(np.int32), a, b, c


def wk_vacuum_form_factors(
    elements: np.ndarray | list[str],
    q: Float[Array, "n_q"],
) -> Float[Array, "n_atoms n_q"]:
    """Gather per-atom vacuum form factors using WK coefficient lookup."""
    q_arr = jnp.asarray(q, dtype=jnp.float32)
    if len(elements) == 0:
        return jnp.zeros((0, int(q_arr.shape[0])), dtype=jnp.float32)
    _, inverse, a, b, c = wk_coefficients_for_elements(elements)
    table = compute_ff_table(a=a, b=b, c=c, q=q_arr)
    return table[jnp.asarray(inverse, dtype=jnp.int32)]


def atomic_volumes_from_elements(elements: np.ndarray | list[str]) -> np.ndarray:
    """Estimate atomic displaced-solvent volumes from VdW radii."""
    symbols = _normalise_elements(elements)
    volumes = np.zeros((symbols.shape[0],), dtype=np.float32)
    for idx, symbol in enumerate(symbols):
        radius = vdw_radius_single(symbol)
        if radius is None:
            raise ValueError(f"Missing VdW radius for element '{symbol}'")
        volumes[idx] = float((4.0 / 3.0) * np.pi * (float(radius) ** 3))
    return volumes


def compute_ff_excl(
    atomic_volumes: Float[Array, "n_atoms"],
    q: Float[Array, "n_q"],
    rho0: float = 0.334,
) -> Float[Array, "n_atoms n_q"]:
    """Compute Fraser Gaussian-sphere excluded-volume form factors."""
    if rho0 <= 0:
        raise ValueError("rho0 must be > 0")
    v = jnp.asarray(atomic_volumes, dtype=jnp.float32)[:, None]
    q_arr = jnp.asarray(q, dtype=jnp.float32)[None, :]
    sphere = (3.0 * v / (4.0 * jnp.pi)) ** (2.0 / 3.0)
    return jnp.asarray(rho0, dtype=jnp.float32) * v * jnp.exp(-sphere * (q_arr**2) / (4.0 * jnp.pi))


def compute_ff_water(
    ff_h: Float[Array, "n_q"],
    ff_o: Float[Array, "n_q"],
    q: Float[Array, "n_q"],
    rho0: float = 0.334,
    v_water: float = _WATER_VOLUME,
) -> Float[Array, "n_q"]:
    """Compute hydration-water form factor: ``2*f_H + f_O - f_excl_water``."""
    if rho0 <= 0:
        raise ValueError("rho0 must be > 0")
    if v_water <= 0:
        raise ValueError("v_water must be > 0")
    q_arr = jnp.asarray(q, dtype=jnp.float32)
    ff_h_arr = jnp.asarray(ff_h, dtype=jnp.float32)
    ff_o_arr = jnp.asarray(ff_o, dtype=jnp.float32)

    sphere = (3.0 * v_water / (4.0 * jnp.pi)) ** (2.0 / 3.0)
    ff_excl_water = jnp.asarray(rho0 * v_water, dtype=jnp.float32) * jnp.exp(
        -sphere * (q_arr**2) / (4.0 * jnp.pi)
    )
    return 2.0 * ff_h_arr + ff_o_arr - ff_excl_water


def wk_water_form_factor(
    q: Float[Array, "n_q"],
    rho0: float = 0.334,
    v_water: float = _WATER_VOLUME,
) -> Float[Array, "n_q"]:
    """Build ``f_water(q)`` from WK hydrogen and oxygen vacuum curves."""
    q_arr = jnp.asarray(q, dtype=jnp.float32)
    ff_h = wk_vacuum_form_factors(["H"], q_arr)[0]
    ff_o = wk_vacuum_form_factors(["O"], q_arr)[0]
    return compute_ff_water(ff_h=ff_h, ff_o=ff_o, q=q_arr, rho0=rho0, v_water=v_water)


__all__ = [
    "atomic_volumes_from_elements",
    "compute_ff_excl",
    "compute_ff_table",
    "compute_ff_water",
    "wk_coefficients_for_elements",
    "wk_vacuum_form_factors",
    "wk_water_form_factor",
]
