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
        c=4.9e-05,
    ),
    "He": _WKEntry(
        a=(0.732354, 0.753896, 0.283819, 0.190003, 0.039139),
        b=(11.553918, 4.595831, 1.546299, 26.463964, 0.377523),
        c=0.000487,
    ),
    "Li": _WKEntry(
        a=(0.974637, 0.150472, 0.811055, 0.262416, 0.790108),
        b=(4.334946, 0.342451, 97.102969, 120.363024, 1.409234),
        c=0.012942,
    ),
    "Be": _WKEntry(
        a=(1.533712, 0.638283, 0.601052, 0.106139, 1.110414),
        b=(42.662078, 0.59542, 99.106501, 0.15134, 1.843093),
        c=0.002511,
    ),
    "B": _WKEntry(
        a=(2.005185, 1.06458, 1.06278, 0.140515, 0.641784),
        b=(23.494069, 1.137894, 61.230975, 0.114886, 0.399036),
        c=0.003112,
    ),
    "C": _WKEntry(
        a=(2.657506, 1.078079, 1.490909, -4.24107, 0.713791),
        b=(14.780758, 0.776775, 42.086842, -0.000294, 0.239535),
        c=4.297983,
    ),
    "N": _WKEntry(
        a=(11.89378, 3.277479, 1.858092, 0.858927, 0.912985),
        b=(0.000158, 10.232723, 30.34469, 0.656065, 0.217287),
        c=-11.804902,
    ),
    "O": _WKEntry(
        a=(2.960427, 2.508818, 0.637853, 0.722838, 1.142756),
        b=(14.182259, 5.936858, 0.112726, 34.958481, 0.39024),
        c=0.027014,
    ),
    "F": _WKEntry(
        a=(3.511943, 2.772244, 0.678385, 0.915159, 1.089261),
        b=(10.687859, 4.380466, 0.093982, 27.255203, 0.313066),
        c=0.032557,
    ),
    "Ne": _WKEntry(
        a=(4.183749, 2.905726, 0.520513, 1.135641, 1.228065),
        b=(0.175457, 3.252536, 0.043295, 21.013909, 0.224952),
        c=0.025576,
    ),
    "Na": _WKEntry(
        a=(4.910127, 3.001785, 1.262067, 1.098938, 0.500991),
        b=(3.281434, 9.119178, 0.102763, 132.013942, 0.405418),
        c=0.079712,
    ),
    "Mg": _WKEntry(
        a=(4.708971, 1.194814, 1.550157, 1.170413, 3.339403),
        b=(4.075207, 108.506079, 0.111516, 48.292407, 1.928171),
        c=0.126842,
    ),
    "Al": _WKEntry(
        a=(4.730796, 2.313951, 1.54198, 1.117564, 3.154754),
        b=(3.620931, 43.051166, 0.09596, 100.932109, 1.555918),
        c=0.139509,
    ),
    "Si": _WKEntry(
        a=(5.275329, 3.191038, 1.511514, 1.356849, 2.519114),
        b=(2.631338, 33.730728, 0.081119, 86.28864, 1.170087),
        c=0.145073,
    ),
    "P": _WKEntry(
        a=(1.950541, 4.14693, 1.49456, 1.522042, 5.729711),
        b=(0.908139, 27.044952, 0.07128, 67.520187, 1.981173),
        c=0.155233,
    ),
    "S": _WKEntry(
        a=(6.372157, 5.154568, 1.473732, 1.635073, 1.209372),
        b=(1.514347, 22.092527, 0.061373, 55.445175, 0.646925),
        c=0.154722,
    ),
    "Cl": _WKEntry(
        a=(1.446071, 6.870609, 6.151801, 1.750347, 0.634168),
        b=(0.052357, 1.193165, 18.343416, 46.398394, 0.401005),
        c=0.146773,
    ),
    "Ar": _WKEntry(
        a=(7.188004, 6.638454, 0.45418, 1.929593, 1.523654),
        b=(0.956221, 15.339877, 15.339862, 39.043824, 0.062409),
        c=0.165954,
    ),
    "K": _WKEntry(
        a=(8.163991, 7.146945, 1.07014, 0.877316, 1.486434),
        b=(12.816323, 0.808945, 210.327009, 39.597651, 0.052821),
        c=0.253614,
    ),
    "Ca": _WKEntry(
        a=(0.593655, 1.477324, 1.436254, 1.182839, 7.11325),
        b=(10.460644, 0.041891, 61.390382, 169.847839, 0.688098),
        c=0.196255,
    ),
    "Sc": _WKEntry(
        a=(1.47656, 1.48727, 1.60018, 9.177463, 7.09975),
        b=(53.13102, 0.03532, 137.31949, 9.09803, 0.6021),
        c=0.15776,
    ),
    "Ti": _WKEntry(
        a=(9.81852, 1.52264, 1.7031, 1.76877, 7.00255),
        b=(8.00187, 0.02976, 39.00542, 120.158, 0.5324),
        c=0.10247,
    ),
    "V": _WKEntry(
        a=(10.47357, 1.5478, 1.9863, 1.86561, 7.05625),
        b=(7.08194, 0.02604, 31.909672, 108.02204, 0.47488),
        c=0.06774,
    ),
    "Cr": _WKEntry(
        a=(11.00706, 1.55547, 2.98529, 1.34705, 7.03477),
        b=(6.36628, 0.02398, 0.244558, 105.7745, 0.42936),
        c=0.06551,
    ),
    "Mn": _WKEntry(
        a=(11.70954, 1.73341, 2.67314, 2.02336, 7.00318),
        b=(5.59712, 0.0178, 21.78841, 9.51791, 0.38305),
        c=-0.14729,
    ),
    "Fe": _WKEntry(
        a=(12.31109, 1.87462, 3.06617, 2.07045, 6.97518),
        b=(5.00941, 0.01446, 10.74304, 2.76707, 0.3465),
        c=-0.30493,
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
