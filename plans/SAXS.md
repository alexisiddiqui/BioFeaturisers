# SAXS Profile Implementation (FoXS-style)

## Overview

The SAXS module computes differentiable small-angle X-ray scattering profiles from
atomic coordinates using the Debye formula with the FoXS six-partial-sum
decomposition. The forward model is fully differentiable with respect to atomic
coordinates and the hydration parameters c₁/c₂, enabling gradient-based structure
refinement against experimental profiles.

The module follows the `featurise → forward → predict` interface pattern: form
factor tables and SASA fractions are computed once at featurisation time, and all
subsequent forward passes operate on raw coordinate arrays.

---

## Mathematical formulation

### Debye formula

The solution X-ray scattering intensity at scattering vector magnitude *q* is:

$$I(q) = \sum_i \sum_j F_i(q) \cdot F_j(q) \cdot \text{sinc}(q \cdot r_{ij})$$

where $r_{ij} = ||\mathbf{r}_i - \mathbf{r}_j||$, $\text{sinc}(x) = \sin(x)/x$, and
$F_i(q)$ is the effective form factor of atom *i*.

This is an O(N²) sum per q-value. For N=5,000 atoms and Q=300 q-values, naive
materialisation of the N×N distance matrix costs 100 MB; the (N, N, Q) sinc tensor
would cost 30 GB. Neither is ever materialised — instead, computation proceeds
in spatial chunks of size B.

### FoXS effective form factors

Each atom's effective form factor decomposes into three physically distinct
contributions:

$$F_i(q) = f_i^{\text{vac}}(q) - c_1 \cdot f_i^{\text{excl}}(q) + c_2 \cdot S_i \cdot f^{\text{water}}(q)$$

where:
- $f^{\text{vac}}(q)$ — vacuum scattering factor (Waasmaier-Kirfel 5-Gaussian)
- $f^{\text{excl}}(q)$ — excluded volume form factor (Fraser Gaussian sphere model)
- $f^{\text{water}}(q)$ — hydration water form factor (shared across all atoms)
- $S_i$ — solvent-accessible surface fraction for atom *i*
- $c_1 \in [0.95, 1.12]$ — excluded volume scaling parameter
- $c_2 \in [0.0, 4.0]$ — hydration layer amplitude

### Six-partial-sum decomposition

Expanding $F_i F_j$ and collecting by powers of c₁ and c₂ yields six
coordinate-dependent partial sums:

$$I(q) = I_{aa} - c_1 I_{ac} + c_1^2 I_{cc} + c_2 I_{as} - c_1 c_2 I_{cs} + c_2^2 I_{ss}$$

where each partial sum has the bilinear form $\mathbf{a}^\top S(q) \mathbf{b}$ with
$S(q)_{ij} = \text{sinc}(q \cdot r_{ij})$. The index labels are:
`a` = vacuum (vac), `c` = excluded volume (excl), `s` = hydration (water/solvent).

The key insight: **the sinc matrix depends only on coordinates, not on c₁ or c₂**.
Fitting c₁/c₂ requires no recomputation of the O(N²) Debye sums — only the
microsecond polynomial recombination.

### Form factor parameterisations

**Vacuum form factors (Waasmaier-Kirfel 1995):** 5-Gaussian + constant per element:

$$f^{\text{vac}}(q) = \sum_{k=1}^{5} a_k \exp(-b_k s^2) + c, \quad s = \frac{q}{4\pi}$$

At $s = 0$, the sum equals the total electron count for the neutral atom.

**Excluded volume (Fraser model):**

$$f^{\text{excl}}(q) = \rho_0 V \exp\!\left(-\left(\frac{3V}{4\pi}\right)^{2/3} \frac{q^2}{4\pi}\right)$$

where $V$ is the atomic volume and $\rho_0 \approx 0.334$ e/Å³ is the bulk solvent
electron density. Since c₁ scales these volumes uniformly, tables can be
precomputed at c₁ = 1 and the scaling absorbed algebraically into the six partial
sums.

**Water form factor:**

$$f^{\text{water}}(q) = 2 f_H(q) + f_O(q) - f^{\text{excl}}_{\text{water}}(q)$$

This is a single q-dependent curve shared across all atoms. Each atom's hydration
contribution is $S_i \cdot f^{\text{water}}(q)$, giving $f^{\text{water}}_{\text{sol},i}$ an
(N, Q) array with rank-1 structure along the atom dimension.

### sinc singularity and diagonal contribution

The i = j diagonal has $r_{ij} = 0$, so $\text{sinc}(0) = 1$ always. The diagonal
contribution is separated analytically before the chunked loop:

$$\text{diag}(q) = \sum_i F_i(q)^2$$

This has trivially clean gradients ($\partial F^2 / \partial F = 2F$) and avoids
the sinc singularity entirely. Off-diagonal blocks use the custom-VJP `safe_sinc`
kernel.

---

## Minimal inputs

### Required per-atom topology (from structure file)

| Field | Shape | Description |
|---|---|---|
| `atom_names` | (N,) str | e.g. "N", "CA", "C", "O", "CB" |
| `element` | (N,) str | element symbols for form factor lookup |
| `res_ids` | (N,) int | residue sequence numbers |
| `chain_ids` | (N,) str | chain identifiers |
| `is_hetatm` | (N,) bool | True for ligand/HETATM atoms |
| `coords` | (N, 3) float32 | atomic coordinates in Å |

### Derived topology features (computed in `featurise`, static thereafter)

| Feature | Shape | Description |
|---|---|---|
| `atom_idx` | (n_sel,) int32 | selected atom indices (post output_index filtering) |
| `ff_vac` | (n_sel, Q) float32 | vacuum form factors |
| `ff_excl` | (n_sel, Q) float32 | excluded volume form factors |
| `ff_water` | (n_sel, Q) float32 | hydration form factors = Si × f_water(q) |
| `solvent_acc` | (n_sel,) float32 | Si fractions from Shrake-Rupley SASA |
| `q_values` | (Q,) float32 | q-grid in Å⁻¹ |

**Note on SASA:** `solvent_acc` (Si) is computed once in `featurise` using biotite's
Lee-Richards Shrake-Rupley algorithm and stored as a static constant. A differentiable
JAX SASA kernel (allowing Si gradients to flow to coordinates) is deferred to a
future milestone.

---

## Data objects

### `SAXSConfig`

```python
@dataclass
class SAXSConfig:
    # Q-grid
    q_min:   float = 0.01   # Å⁻¹
    q_max:   float = 0.50   # Å⁻¹
    n_q:     int   = 300

    # FoXS hydration model
    c1:           float = 1.0    # excluded volume scaling
    c2:           float = 0.0    # hydration layer amplitude
    fit_c1_c2:    bool  = True   # grid-search c1/c2 in predict
    c1_range:     tuple = (0.95, 1.12)
    c2_range:     tuple = (0.0, 4.0)
    c1_steps:     int   = 18
    c2_steps:     int   = 17
    rho0:         float = 0.334  # solvent electron density (e/Å³)

    # Scattering factor table
    ff_table: str = "waasmaier_kirfel"   # or "cromer_mann"

    # Compute
    chunk_size:  int = 512    # B in chunked Debye sum
    batch_size:  int = 4      # lax.map batch for trajectory processing

    # Atom selection
    include_chains:  list[str] | None = None   # None = all chains
    exclude_chains:  list[str] | None = None
    include_hetatm:  bool = False
```

### `SAXSFeatures`

```python
@dataclass
class SAXSFeatures:
    topology:      MinimalTopology
    output_index:  OutputIndex     # controls which atoms contribute to I(q)

    # Atom selection (post output_index filtering)
    atom_idx:      np.ndarray      # (n_sel,) int32

    # Form factor tables — precomputed at c1=1 (scaling handled algebraically)
    ff_vac:        np.ndarray      # (n_sel, Q) float32
    ff_excl:       np.ndarray      # (n_sel, Q) float32
    ff_water:      np.ndarray      # (n_sel, Q) float32 — Si * f_water(q)
    solvent_acc:   np.ndarray      # (n_sel,) float32 — Si fractions (static)

    q_values:      np.ndarray      # (Q,) float32

    # Metadata
    chain_ids:     np.ndarray      # (n_sel,) str — for provenance reporting

    def save(self, prefix: str) -> None: ...
    @classmethod
    def load(cls, prefix: str) -> "SAXSFeatures": ...
```

---

## JAX kernels (`saxs/forward.py`)

### safe_sinc — the most critical primitive

`sinc(qr) = sin(qr)/qr` has a removable singularity at `qr = 0`. JAX evaluates
both branches of `jnp.where` during autodiff, so a NaN/Inf in the unmasked branch
poisons the gradient even if the forward output is correct. The canonical fix uses
a **custom VJP** with double-where on the input:

```python
@jax.custom_vjp
def safe_sinc(qr: jax.Array) -> jax.Array:
    """
    sinc(qr) = sin(qr)/qr  with Taylor expansion at the origin.
    Forward:
      qr > 1e-8  →  sin(qr)/qr
      qr ≤ 1e-8  →  1 - qr²/6 + qr⁴/120   (Taylor)
    """
    safe_qr = jnp.where(qr > 1e-8, qr, 1.0)   # safe INPUT — never feeds NaN to sin/div
    return jnp.where(
        qr > 1e-8,
        jnp.sin(safe_qr) / safe_qr,
        1.0 - qr**2 / 6.0 + qr**4 / 120.0,
    )

def _safe_sinc_fwd(qr):
    y = safe_sinc(qr)
    return y, (qr, y)    # store both — avoids recomputing sinc in backward

def _safe_sinc_bwd(res, g):
    qr, y    = res
    safe_qr  = jnp.where(qr > 1e-8, qr, 1.0)
    # sinc'(x) = (cos(x) - sinc(x)) / x;  Taylor at origin: -x/3 + x³/30
    dsinc = jnp.where(
        qr > 1e-8,
        (jnp.cos(safe_qr) - y) / safe_qr,
        -qr / 3.0 + qr**3 / 30.0,
    )
    return (g * dsinc,)

safe_sinc.defvjp(_safe_sinc_fwd, _safe_sinc_bwd)
```

The custom VJP avoids accumulating an autodiff graph through the sin/division chain
and analytically computes the correct derivative at the origin (gradient = 0 at
qr = 0, since the gradient of sinc at the origin is 0).

### Block distance computation

For symmetric SAXS blocks, broadcasting is preferred over the matmul identity
because atoms within the same molecule have similar coordinate magnitudes, making
`||a||² - 2a·b + ||b||²` susceptible to catastrophic cancellation at small
separations:

```python
def dist_matrix_block(coords_i, coords_j):
    """Returns dist_sq (not dist) — diagonal may contain exact zeros."""
    diff = coords_i[:, None, :] - coords_j[None, :, :]   # (B_i, B_j, 3)
    return jnp.sum(diff**2, axis=-1)                       # (B_i, B_j)

def dist_from_sq_block(dist_sq):
    """Double-where sqrt: dist=0 exactly on diagonal, finite gradient."""
    is_zero = dist_sq <= 0.0
    safe_sq = jnp.where(is_zero, 1.0, dist_sq)   # safe INPUT before sqrt
    return jnp.where(is_zero, 0.0, jnp.sqrt(safe_sq))
```

The diagonal (`dist = 0`, `sinc = 1`) is handled separately via
`diagonal_self_pairs` — the chunk loop processes only off-diagonal blocks.

### Six-partial-sum block kernel

Within each B×B chunk, the sinc kernel is computed once and contracted with three
form-factor vectors simultaneously:

```python
def six_partial_sums_block(
    sinc_block: jax.Array,   # (B_i, B_j, Q)
    fv_i, fv_j,              # vacuum form factors (B, Q)
    fe_i, fe_j,              # excluded volume form factors (B, Q)
    fs_i, fs_j,              # hydration form factors (B, Q)
) -> jax.Array:              # (6, Q)
    """
    Three sinc-weighted sums (the expensive step) — then six cheap contractions.
    Memory overhead vs. single sum: three extra (B, Q) vectors.
    """
    # Sinc-weighted form factor sums: w_x[i,q] = Σ_j ff_x[j,q] * sinc[i,j,q]
    w_v = jnp.einsum('ijq,jq->iq', sinc_block, fv_j)   # (B, Q)
    w_e = jnp.einsum('ijq,jq->iq', sinc_block, fe_j)
    w_s = jnp.einsum('ijq,jq->iq', sinc_block, fs_j)

    # Six bilinear contractions
    Iaa = jnp.sum(fv_i * w_v, axis=0)                           # (Q,)
    Icc = jnp.sum(fe_i * w_e, axis=0)
    Iss = jnp.sum(fs_i * w_s, axis=0)
    Iac = jnp.sum(fv_i * w_e + fe_i * w_v, axis=0)
    Ias = jnp.sum(fv_i * w_s + fs_i * w_v, axis=0)
    Ics = jnp.sum(fe_i * w_s + fs_i * w_e, axis=0)

    return jnp.stack([Iaa, Icc, Iss, Iac, Ias, Ics])   # (6, Q)
```

XLA fuses the `einsum` multiply-reduce chain into a single GPU reduction kernel.

### Chunked double-scan Debye sum

```python
@partial(jax.jit, static_argnames=["chunk_size"])
def saxs_six_partials(
    coords:     jax.Array,       # (N_atoms, 3) — full coordinate array
    features:   SAXSFeatures,
    chunk_size: int = 512,
) -> jax.Array:                  # (6, Q)
    """
    Double lax.scan over spatial chunks.
    Peak memory per block: O(B² × Q).
    Never materialises the full (N, N) or (N, N, Q) arrays.
    """
    # Gather selected atoms
    coords_sel = coords[features.atom_idx]    # (n_sel, 3)
    fv  = features.ff_vac                     # (n_sel, Q)
    fe  = features.ff_excl
    fs  = features.ff_water

    N, Q = fv.shape
    B    = chunk_size

    # Pad to multiple of B (zero ff → zero contribution)
    pad_n    = (-N) % B
    coords_p = jnp.pad(coords_sel, ((0, pad_n), (0, 0)))
    fv_p     = jnp.pad(fv,  ((0, pad_n), (0, 0)))
    fe_p     = jnp.pad(fe,  ((0, pad_n), (0, 0)))
    fs_p     = jnp.pad(fs,  ((0, pad_n), (0, 0)))

    n_ch       = coords_p.shape[0] // B
    coords_ch  = coords_p.reshape(n_ch, B, 3)
    fv_ch, fe_ch, fs_ch = (x.reshape(n_ch, B, Q) for x in (fv_p, fe_p, fs_p))

    # Diagonal contribution: Σ_i F_i(q)² (sinc(0) = 1 always)
    diag = (
        jnp.sum(fv**2, axis=0)
        - 2.0 * jnp.sum(fv * fe, axis=0)   # cross terms for Iac at diagonal
        + jnp.sum(fe**2, axis=0)
        # ... all six diagonal contributions computed analytically
    )
    # Simplified: compute diagonal_self_pairs for each of the six partial sums
    diag_partials = jnp.stack([
        jnp.sum(fv**2, axis=0),                          # Iaa diagonal
        jnp.sum(fe**2, axis=0),                          # Icc diagonal
        jnp.sum(fs**2, axis=0),                          # Iss diagonal
        2.0 * jnp.sum(fv * fe, axis=0),                  # Iac diagonal
        2.0 * jnp.sum(fv * fs, axis=0),                  # Ias diagonal
        2.0 * jnp.sum(fe * fs, axis=0),                  # Ics diagonal
    ])   # (6, Q)

    def outer_scan(carry, i):
        def inner_scan(carry_in, j):
            ci, cj = coords_ch[i], coords_ch[j]
            dist_sq = dist_matrix_block(ci, cj)    # (B, B)

            # Skip exact diagonal blocks (handled analytically above)
            # For i != j blocks: dist_sq > 0 always (different atom sets)
            # For i == j blocks: diagonal entries are zero — use dist_from_sq_block
            dist    = dist_from_sq_block(dist_sq)  # (B, B)
            qr      = features.q_values[None, None, :] * dist[:, :, None]  # (B, B, Q)
            sinc_bl = safe_sinc(qr)                # (B, B, Q)

            block_contrib = six_partial_sums_block(
                sinc_bl,
                fv_ch[i], fv_ch[j],
                fe_ch[i], fe_ch[j],
                fs_ch[i], fs_ch[j],
            )   # (6, Q)

            # Weight: 2 for off-diagonal (exploits Σ_ij = Σ_i<j + Σ_i>j + diag);
            # 1 for diagonal (but diagonal handled analytically, so subtract diag here)
            is_diag = (i == j)
            # On-diagonal block: block_contrib includes the i==j terms,
            # which we've already counted in diag_partials.
            # Subtract the diagonal atoms' self-pair contribution from this block.
            diag_in_block = jnp.stack([
                jnp.sum(fv_ch[i]**2, axis=0),
                jnp.sum(fe_ch[i]**2, axis=0),
                jnp.sum(fs_ch[i]**2, axis=0),
                2.0 * jnp.sum(fv_ch[i] * fe_ch[i], axis=0),
                2.0 * jnp.sum(fv_ch[i] * fs_ch[i], axis=0),
                2.0 * jnp.sum(fe_ch[i] * fs_ch[i], axis=0),
            ])
            weight = jnp.where(is_diag, 1.0, 2.0)
            off_diag_contrib = jnp.where(
                is_diag,
                block_contrib - diag_in_block,   # remove self-pairs already counted
                block_contrib,
            )
            return carry_in + weight * off_diag_contrib, None

        carry, _ = jax.lax.scan(inner_scan, carry, jnp.arange(n_ch))
        return carry, None

    off_diag_partials, _ = jax.lax.scan(
        outer_scan, jnp.zeros((6, Q)), jnp.arange(n_ch)
    )
    return diag_partials + off_diag_partials   # (6, Q)
```

### Polynomial recombination

```python
def saxs_combine(
    partials: jax.Array,   # (6, Q) — [Iaa, Icc, Iss, Iac, Ias, Ics]
    c1: float,
    c2: float,
) -> jax.Array:            # (Q,) I(q)
    """
    I(q) = Iaa - c1·Iac + c1²·Icc + c2·Ias - c1·c2·Ics + c2²·Iss

    Analytic gradients:
        ∂I/∂c1 = -Iac + 2c1·Icc - c2·Ics
        ∂I/∂c2 =  Ias - c1·Ics  + 2c2·Iss
    """
    Iaa, Icc, Iss, Iac, Ias, Ics = partials
    return Iaa - c1*Iac + c1**2*Icc + c2*Ias - c1*c2*Ics + c2**2*Iss
```

### Full forward function

```python
@partial(jax.jit, static_argnames=["chunk_size"])
def saxs_forward(
    coords:     jax.Array,
    features:   SAXSFeatures,
    c1:         float = 1.0,
    c2:         float = 0.0,
    chunk_size: int   = 512,
) -> jax.Array:   # (Q,) I(q)
    partials = saxs_six_partials(coords, features, chunk_size=chunk_size)
    return saxs_combine(partials, c1, c2)
```

---

## c₁/c₂ grid search and fitting

Because c₁ and c₂ enter only through the polynomial recombination, fitting them
requires no recomputation of the O(N²) Debye sums:

```python
def fit_c1_c2(
    partials:  jax.Array,   # (6, Q) — precomputed from saxs_six_partials
    I_exp:     jax.Array,   # (Q,) experimental intensities
    sigma:     jax.Array,   # (Q,) experimental uncertainties
    config:    SAXSConfig,
) -> tuple[float, float, float]:
    """
    Grid search over (c1, c2), analytic scale factor per grid point.
    Returns (c1_opt, c2_opt, chi2_min).
    Runs in microseconds once partials are computed.
    """
    c1_grid = jnp.linspace(*config.c1_range, config.c1_steps)
    c2_grid = jnp.linspace(*config.c2_range, config.c2_steps)

    def chi2_for(c1, c2):
        I_calc = saxs_combine(partials, c1, c2)
        # Analytic scale factor: k = (I_calc·I_exp/σ²) / (I_calc²/σ²)
        k = jnp.sum(I_calc * I_exp / sigma**2) / jnp.sum(I_calc**2 / sigma**2)
        residuals = (k * I_calc - I_exp) / sigma
        return jnp.sum(residuals**2) / (len(I_exp) - 1)

    chi2_grid = jax.vmap(
        jax.vmap(chi2_for, (None, 0)), (0, None)
    )(c1_grid, c2_grid)   # (n_c1, n_c2)

    flat_idx = jnp.argmin(chi2_grid)
    i, j = jnp.unravel_index(flat_idx, chi2_grid.shape)
    return float(c1_grid[i]), float(c2_grid[j]), float(chi2_grid[i, j])
```

---

## Form factor tables (`saxs/form_factors.py`)

### Waasmaier-Kirfel 5-Gaussian evaluation

```python
def compute_ff_table(
    a:  np.ndarray,   # (T, 5)  Gaussian amplitudes per element type
    b:  np.ndarray,   # (T, 5)  Gaussian exponents
    c:  np.ndarray,   # (T,)    constant term
    q:  jax.Array,   # (Q,)
) -> jax.Array:       # (T, Q)
    s2       = (q / (4.0 * jnp.pi))**2                    # (Q,)
    exponents = -b[:, :, None] * s2[None, None, :]         # (T, 5, Q)
    return jnp.sum(a[:, :, None] * jnp.exp(exponents), axis=1) + c[:, None]

# Per-atom form factors via integer gather (JIT-friendly)
ff_vac_per_atom = ff_table[atom_type_indices]   # (N, Q)
```

Element-type dispatch via `ff_table[atom_types]` is a single vectorised gather
operation — vastly superior to `lax.cond/switch` per atom.

### Excluded volume form factors

```python
def compute_ff_excl(
    atomic_volumes: jax.Array,   # (N,) Å³ — per atom type
    q:              jax.Array,   # (Q,)
    rho0:           float = 0.334,
) -> jax.Array:                  # (N, Q)
    """Fraser Gaussian sphere model."""
    V = atomic_volumes[:, None]   # (N, 1)
    sphere_factor = (3.0 * V / (4.0 * jnp.pi))**(2.0/3.0)
    return rho0 * V * jnp.exp(-sphere_factor * q[None, :]**2 / (4.0 * jnp.pi))
```

### Water form factor

```python
def compute_ff_water(
    ff_H:  jax.Array,   # (Q,) hydrogen vacuum form factor
    ff_O:  jax.Array,   # (Q,) oxygen vacuum form factor
    q:     jax.Array,   # (Q,)
    rho0:  float = 0.334,
    V_water: float = 29.9,   # Å³ — volume of one water molecule
) -> jax.Array:              # (Q,)
    """f_water = 2·f_H + f_O - f_excl_water"""
    ff_excl_water = rho0 * V_water * jnp.exp(
        -(3.0*V_water/(4.0*jnp.pi))**(2.0/3.0) * q**2 / (4.0*jnp.pi)
    )
    return 2.0 * ff_H + ff_O - ff_excl_water

# Per-atom hydration form factor: (N, Q)
ff_water_per_atom = solvent_acc[:, None] * ff_water[None, :]   # S_i * f_water(q)
```

---

## Trajectory processing

```python
@jax.jit
def saxs_trajectory(
    trajectory:  jax.Array,    # (T, N_atoms, 3)
    features:    SAXSFeatures,
    c1:          float = 1.0,
    c2:          float = 0.0,
    weights:     jax.Array | None = None,   # (T,) normalised; None = uniform
    batch_size:  int   = 4,
    chunk_size:  int   = 512,
) -> jax.Array:                # (Q,) ensemble-averaged I(q)
    """
    Uses lax.map(batch_size) + jax.checkpoint for bounded memory.
    For T=1000, N=5000, Q=300 with batch_size=4:
      peak memory ≈ 4 × ~300 MB + form factors ≈ 1.3 GB.
    """
    @jax.checkpoint
    def per_frame(coords_t):
        return saxs_six_partials(coords_t, features, chunk_size=chunk_size)

    all_partials = jax.lax.map(per_frame, trajectory, batch_size=batch_size)
    # all_partials: (T, 6, Q)

    if weights is None:
        mean_partials = jnp.mean(all_partials, axis=0)
    else:
        mean_partials = jnp.sum(weights[:, None, None] * all_partials, axis=0)

    return saxs_combine(mean_partials, c1, c2)   # (Q,)
```

**Note:** for ensemble refinement with c₁/c₂ fitting, compute and cache
`all_partials` first, then run the grid search over cached partials — both the
trajectory average and the c₁/c₂ optimisation are free once partials are in hand.

---

## JIT compilation and shape strategy

### Padding strategy

Zero form factors naturally eliminate padding-atom contributions through the physics
— no explicit mask needed:

```python
# Pad to multiple of chunk_size
padded_ff  = jnp.zeros((max_atoms, Q)).at[:N].set(form_factors)
padded_xyz = jnp.zeros((max_atoms, 3)).at[:N].set(coords)
# Padding atom FF = 0 → F_i·F_j = 0 for any pair involving padding → zero contribution
# Gradient w.r.t. padding atom coordinates = 0 → clean backprop
```

### Power-of-2 atom buckets

```python
BUCKETS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

def get_bucket(n: int) -> int:
    for b in BUCKETS:
        if n <= b: return b
    return BUCKETS[-1]
```

At most 9 JIT compilations are cached per session.

### JIT boundary and coordinate buffer donation

```python
@partial(jax.jit, static_argnames=["chunk_size"], donate_argnums=(0,))
def saxs_loss(
    coords:    jax.Array,   # (N, 3) — XLA may recycle this buffer (donate_argnums)
    features:  SAXSFeatures,
    I_exp:     jax.Array,   # (Q,)
    sigma:     jax.Array,   # (Q,)
    c1:        float,
    c2:        float,
    chunk_size: int = 512,
) -> jax.Array:             # scalar χ²
    I_calc = saxs_forward(coords, features, c1, c2, chunk_size=chunk_size)
    scale  = jnp.sum(I_calc * I_exp / sigma**2) / jnp.sum(I_calc**2 / sigma**2)
    residuals = (scale * I_calc - I_exp) / sigma
    return jnp.sum(residuals**2) / (len(I_exp) - 1)
```

Do **not** use `static_argnums` for atom-type arrays — integer arrays trigger
recompilation on any change. Pass precomputed form factor tables as traced float
arrays instead.

---

## Block size selection

Peak per-block memory = B² × Q × 4 bytes:

| B | Q=100 | Q=300 | Blocks for N=50K | Recommended GPU |
|---|-------|-------|-----------------|-----------------|
| 256 | 25 MB | 75 MB | 38,025 | Consumer 4-8 GB |
| 512 | 100 MB | 300 MB | 9,604 | Consumer 8-16 GB |
| 1024 | 400 MB | 1.2 GB | 2,401 | A100 80 GB |
| 2048 | 1.6 GB | 4.8 GB | 625 | A100 (single-block) |

For large Q, batch q-values in groups of 32–64 and loop over q-batches to reduce
peak memory further.

---

## Precision

**Float32 is sufficient** for typical SAXS. The sinc argument `qr` can reach ~250
for `q_max = 0.5 Å⁻¹` and `r_max = 500 Å`, where float32 retains ~4 digits of
precision for `sin(250)` — acceptable since large-distance sinc terms contribute
negligibly to the total.

The chunked approach provides hierarchical summation, limiting accumulation error:
for N=5,000 with B=512, ~100 partial sums are accumulated rather than 25M terms.
Relative error drops from ~0.6 (catastrophic, naive) to ~1.2×10⁻⁵ (acceptable).

**Float64** can be enabled for sensitive gradient-based refinement:
```python
jax.config.update("jax_enable_x64", True)
I_accum = jnp.zeros(Q, dtype=jnp.float64)
```

Float64 is 2× slower on consumer GPUs but runs at full speed on A100/H100.
**bfloat16 and float16 are not viable** — insufficient mantissa bits for sin(qr)
at large arguments.

---

## OutputIndex for SAXS

The `OutputIndex` controls which atoms contribute to I(q):

| Use case | Configuration |
|---|---|
| Standard protein (no ligands) | `include_hetatm=False` (default) |
| Include ligands in scattering | `include_hetatm=True` |
| Single-chain from complex | `include_chains=["A"]` |
| Exclude disordered region | `custom_atom_mask` |

Unlike HDX, the SAXS output is always a single `I(q)` curve — `output_index`
determines which atoms *contribute*, not which residues *appear in a table*.
Chain/selection metadata is recorded in `SAXSFeatures` for provenance only.

---

## Public interfaces

### `featurise`

```python
features: SAXSFeatures = saxs.featurise(
    structure,            # biotite AtomArray, PDB path, mmCIF path, or .xtc
    config=SAXSConfig(),
    output_index=None,    # None = all protein chains, no HETATM
)
features.save("my_protein")   # my_protein_features.npz + my_protein_topology.json
```

### `forward`

```python
features = SAXSFeatures.load("my_protein")

# Single structure
I_q = saxs.forward(coords, features, config)           # (Q,)

# With explicit c1/c2
I_q = saxs.forward(coords, features, config, c1=1.05, c2=2.1)

# Trajectory — uniform ensemble mean
I_q = saxs.forward(coords_traj, features, config)      # coords_traj: (T, N, 3)

# Trajectory — weighted ensemble (BME / MaxEnt reweighting)
I_q = saxs.forward(coords_traj, features, config, weights=w)
```

### `predict`

```python
I_q = saxs.predict("protein.pdb", config=SAXSConfig())

# Fit c1/c2 against experimental data
I_q, chi2, c1_opt, c2_opt = saxs.predict(
    "protein.pdb",
    config=SAXSConfig(fit_c1_c2=True),
    I_exp=I_experimental,
    sigma=sigma_experimental,
)
```

---

## Output serialisation

```
{prefix}_saxs_output.npz    # keys: I_q, q_values, [partials if requested]
{prefix}_saxs_index.json    # chain_ids, atom_counts, c1_used, c2_used
```

All float arrays stored as float32. The `partials` array (6, Q) can optionally be
saved to enable post-hoc c₁/c₂ grid search without rerunning the Debye sum.

---

## Performance expectations

| System | Per-profile time | Notes |
|---|---|---|
| N=1,000, Q=100 | Sub-ms | No chunking needed, ~4 MB distance block |
| N=5,000, Q=300 | 10–50 ms | ~100 block operations at B=512 |
| N=50,000, Q=300 | 1–5 s | ~10,000 block operations at B=512 |

Estimates for A100 (80 GB). The O(N²) scaling is the fundamental bottleneck.
For systems above ~20,000 atoms, consider Cα-only coarse-graining (reduces N by
10–20×) before the full-atom Debye sum.

---

## Validation target

Compare `I(q)` against CRYSOL or FoXS output on the same PDB with default
parameters. Expected: χ² < 1.1 on a noise-free theoretical profile comparison
(testing the implementation, not the model).

Key numerical invariants:

| Test | Expected |
|---|---|
| `safe_sinc(0.0)` | 1.0 |
| `jax.grad(safe_sinc)(0.0)` | 0.0 (not NaN) |
| `dist_from_sq_block(jnp.zeros((4,4)))` | `jnp.zeros((4,4))` |
| grad of `dist_from_sq_block` at zero | finite (not NaN) |
| `saxs_combine(partials, 1.0, 0.0)` | equals FoXS vacuum-only result |
| Padding atoms (zero FF) | zero contribution to I(q) |
| `jax.grad` of χ² scalar loss through `saxs_forward` | no NaN |
| I(q) profile on 1UBQ vs FoXS default | χ² < 1.1 |