# HDX-MS Protection Factor Implementation

## Overview

The HDX module computes differentiable backbone amide hydrogen-deuterium exchange
protection factors from atomic coordinates using the Best-Vendruscolo (BV) model
with the Wan et al. JCTC 2020 sigmoid switching function. It integrates with
HDXrate for intrinsic exchange rates and supports single structures, trajectory
ensembles, and weighted ensembles.

The module follows the `featurise → forward → predict` interface pattern: topology
parsing runs once at featurisation time, and all subsequent forward passes operate
on raw coordinate arrays without re-reading structure files.

---

## Mathematical formulation

### Best-Vendruscolo contact counts

For each exchangeable backbone amide residue *i*, two contact counts are computed:

**Heavy-atom contacts (Nc):** counts all heavy atoms *j* within a distance cutoff
around the amide nitrogen:

$$N_c^i = \sum_j \frac{\exp(-b(r_{ij} - x_c))}{1 + \exp(-b(r_{ij} - x_c))} \cdot \text{mask}_c[i,j]$$

which is identically `jax.nn.sigmoid(b * (x_c - r_ij))`.

**Hydrogen-bond contacts (Nh):** counts backbone carbonyl oxygens *j* within a
short distance cutoff around the amide hydrogen:

$$N_h^i = \sum_j \frac{\exp(-b(r_{ij} - x_h))}{1 + \exp(-b(r_{ij} - x_h))} \cdot \text{mask}_h[i,j]$$

### Protection factor (Wan et al. form)

$$\ln(\text{PF}_i) = \beta_0 + \beta_c \cdot N_c^i + \beta_h \cdot N_h^i$$

The intercept term β₀ compensates for correlation between Nc and Nh — when a
residue is buried (high Nc) it tends to also be hydrogen-bonded (high Nh). Setting
β₀ = 0 recovers the original BV model exactly.

Default parameters: β_c = 0.35, β_h = 2.0, β₀ = 0.0, x_c = 6.5 Å, x_h = 2.4 Å,
b = 10.0 Å⁻¹.

### Deuterium uptake

When HDXrate is enabled, per-peptide deuterium uptake at timepoints **t** is:

$$D(t) = \sum_{k \in \text{peptide}} \text{can\_exchange}_k \cdot \left(1 - \exp(-k_{\text{int},k} \cdot e^{-\ln(\text{PF}_k)} \cdot t)\right)$$

`kint` values are static topology features (not differentiated); gradients flow only
through `ln_Pf`.

### Sigmoid-tanh equivalence

The Wan et al. sigmoid and tanh switching functions are numerically identical:

$$\text{sigmoid}(b \cdot \Delta r) \equiv \frac{1}{2}\left(1 + \tanh\left(\frac{b \cdot \Delta r}{2}\right)\right)$$

so `sigmoid_switch(b)` ≡ `tanh_switch(k = b/2)`. Use `sigmoid_switch` when
comparing against published Wan et al. parameter grids where `b` is reported
directly.

---

## Minimal inputs

### Required per-atom topology (from structure file)

| Field | Shape | Description |
|---|---|---|
| `atom_names` | (N,) str | e.g. "N", "CA", "C", "O", "CB" |
| `res_names` | (N,) str | three-letter residue codes |
| `res_ids` | (N,) int | residue sequence numbers |
| `chain_ids` | (N,) str | chain identifiers |
| `element` | (N,) str | element symbols for heavy-atom classification |
| `is_hetatm` | (N,) bool | True for ligand/HETATM atoms |
| `coords` | (N, 3) float32 | atomic coordinates in Å |

### Derived topology features (computed in `featurise`, static thereafter)

| Feature | Shape | Description |
|---|---|---|
| `amide_N_idx` | (n_res,) int32 | indices of backbone amide N atoms |
| `amide_H_idx` | (n_res,) int32 | indices of backbone amide H atoms (analytically placed if absent) |
| `heavy_atom_idx` | (n_heavy,) int32 | all heavy atom indices for Nc environment |
| `backbone_O_idx` | (n_bb_O,) int32 | backbone O indices for Nh environment |
| `excl_mask_c` | (n_res, n_heavy) float32 | sequence-exclusion + chain-boundary mask for Nc |
| `excl_mask_h` | (n_res, n_bb_O) float32 | sequence-exclusion + chain-boundary mask for Nh |
| `kint` | (n_res,) float32 or None | intrinsic exchange rates (s⁻¹); np.nan for non-exchangeable |
| `can_exchange` | (n_res,) bool | False for Pro, N-term, disulfide Cys |

### Non-exchangeable residues

The following residues are excluded from probe arrays and output:

| Type | Reason | kint stored as |
|---|---|---|
| N-terminal residue (per chain) | no amide H | `np.nan` |
| Proline | no backbone NH | `np.nan` |
| Disulfide-bonded Cys (default) | NH not solvent-accessible | `np.nan` |

`np.nan` unambiguously distinguishes "not computed" from a genuine near-zero rate.

---

## Exclusion mask construction

The sequence-separation exclusion mask encodes which probe–environment pairs are
valid contacts. Cross-chain contacts are always allowed (never excluded by seq-sep
rules); only same-chain pairs within `min_sep` residues are excluded.

```python
def build_exclusion_mask(
    probe_resids:    np.ndarray,   # (n_probe,) int
    probe_chain_ids: np.ndarray,   # (n_probe,) str
    env_resids:      np.ndarray,   # (n_env,) int
    env_chain_ids:   np.ndarray,   # (n_env,) str
    min_sep:         int = 2,
) -> np.ndarray:                   # (n_probe, n_env) float32
    """
    mask[i,j] = 1.0  iff the contact is valid:
      - different chains, OR
      - same chain AND |resid_i - resid_j| > min_sep
    Padding atoms (chain_id == -1) always masked out (mask = 0.0).
    """
    same_chain = (probe_chain_ids[:, None] == env_chain_ids[None, :])
    seq_sep    = np.abs(probe_resids[:, None] - env_resids[None, :])
    too_close  = same_chain & (seq_sep <= min_sep)
    is_padding = (env_chain_ids[None, :] == "-1")
    return (~too_close & ~is_padding).astype(np.float32)
```

For multi-chain complexes, concatenate all chains into flat arrays before calling
this function. The resulting `(n_probe_total, n_env_total)` mask handles all
cross-chain and intra-chain pairs without per-chain logic inside the hot loop.

---

## Amide H placement

X-ray structures often lack hydrogen atoms. When amide H is absent, coordinates are
placed analytically during `featurise` using ideal peptide geometry:

1. For residue *i*, collect N(i), CA(i), C(i-1) positions.
2. Compute the bisector of the N–CA and N–C(i-1) bond vectors.
3. Place H at N + 1.01 Å along the bisector (planar peptide constraint).

This is computed once in biotite before any JAX code runs and stored as
`amide_H_idx` pointing to synthetically appended atoms in the coordinate array.
The forward pass treats placed H atoms identically to observed H atoms.

---

## HDXrate integration

HDXrate computes per-residue intrinsic exchange rates from sequence, temperature,
and pH. The critical constraint is that each chain must be processed independently
since each chain has its own N-terminus (which receives rate 0.0):

```python
# HDXrate API
from hdxrate import k_int_from_sequence
# k_int_from_sequence(sequence: str, temperature: float, pH: float)
# → np.ndarray shape (len(sequence),), units s⁻¹
# index 0 (N-terminal residue of the supplied sequence) → 0.0
# Proline residues                                       → 0.0

# CORRECT: one call per chain
for chain_id in sorted(set(topology.chain_ids)):
    chain_seq = get_chain_one_letter_sequence(topology, chain_id)
    rates = k_int_from_sequence(chain_seq, temperature, pH)
    # rates[0] == 0.0 for THIS chain's N-terminus

# WRONG: concatenating chains before calling HDXrate
# combined_seq = chain_A_seq + chain_B_seq   ← chain B N-term gets non-zero rate
# rates = k_int_from_sequence(combined_seq, T, pH)  ← INCORRECT
```

HDXrate returns 0.0 for Pro and N-terminal residues internally; the implementation
replaces these with `np.nan` to distinguish "non-exchangeable" from "genuine
near-zero rate" in the output index JSON.

---

## Data objects

### `HDXConfig`

```python
@dataclass
class HDXConfig:
    # BV model parameters
    beta_c:          float = 0.35     # weight for heavy-atom contacts (Nc)
    beta_h:          float = 2.0      # weight for H-bond contacts (Nh)
    beta_0:          float = 0.0      # Wan et al. cooperativity intercept; 0.0 = BV
    cutoff_c:        float = 6.5      # Å — Nc heavy-atom cutoff
    cutoff_h:        float = 2.4      # Å — Nh H-bond cutoff
    steepness:       float = 10.0     # b (Å⁻¹); sigmoid sharpness
    seq_sep_min:     int   = 2        # minimum seq separation for valid contact
    intrachain_only: bool  = False    # if True, cross-chain contacts excluded

    # Topology options
    include_hetatm:      bool = False  # ligand heavy atoms contribute to Nc
    disulfide_exchange:  bool = False  # treat disulfide Cys as exchangeable

    # HDXrate integration
    use_hdxrate:  bool  = False
    hdxrate_pH:   float = 7.0
    hdxrate_temp: float = 298.15      # Kelvin
    timepoints:   list[float] = field(default_factory=list)  # seconds

    # Compute
    chunk_size:   int = 0   # 0 = auto dense; >0 = chunked lax.scan fallback
    batch_size:   int = 8   # lax.map batch for trajectory processing
```

### `HDXFeatures`

Serialisable static feature object. Computed once per protein topology; saved to
`{prefix}_features.npz` + `{prefix}_topology.json`. Reload with
`HDXFeatures.load(prefix)` — no biotite required at load time.

```python
@dataclass
class HDXFeatures:
    topology:        MinimalTopology

    # Probe atom indices (into the full coordinate array)
    amide_N_idx:     np.ndarray    # (n_res_exchangeable,) int32
    amide_H_idx:     np.ndarray    # (n_res_exchangeable,) int32

    # Environment atom indices
    heavy_atom_idx:  np.ndarray    # (n_heavy,) int32   — for Nc
    backbone_O_idx:  np.ndarray    # (n_bb_O,)  int32   — for Nh

    # Exclusion masks (float32, shape fixed at featurise time)
    excl_mask_c:     np.ndarray    # (n_res_exchangeable, n_heavy)
    excl_mask_h:     np.ndarray    # (n_res_exchangeable, n_bb_O)

    # Residue metadata — aligned to output_index.output_res_idx
    res_keys:        np.ndarray    # (n_out,) str   — "A:42" canonical keys
    res_names:       np.ndarray    # (n_out,) str   — three-letter codes
    can_exchange:    np.ndarray    # (n_out,) bool

    # Optional intrinsic exchange rates (requires use_hdxrate=True)
    # np.nan for non-exchangeable residues; aligned to res_keys
    kint:            np.ndarray | None   # (n_out,) float32, units s⁻¹

    def save(self, prefix: str) -> None: ...
    @classmethod
    def load(cls, prefix: str) -> "HDXFeatures": ...
```

Output arrays are always aligned to `features.res_keys`. Non-exchangeable residues
(Pro, N-terminal, disulfide Cys) are absent from probe arrays, not zero-padded;
`res_keys` provides canonical `"chain:resid"` keys for alignment to experimental
peptide coverage maps.

---

## JAX kernels (`hdx/forward.py`)

### Core contact calculation

```python
@partial(jax.jit, static_argnames=["b", "x_c", "x_h", "chunk_size"])
def hdx_forward(
    coords:      jax.Array,     # (N_atoms, 3)
    features:    HDXFeatures,
    config:      HDXConfig,
) -> dict[str, jax.Array]:
    """
    Returns: {"Nc": (n_res,), "Nh": (n_res,), "ln_Pf": (n_res,)}
    All arrays aligned to features.res_keys.
    """
    Nc, Nh = bv_contact_counts(
        coords,
        features.amide_N_idx,
        features.amide_H_idx,
        features.heavy_atom_idx,
        features.backbone_O_idx,
        features.excl_mask_c,
        features.excl_mask_h,
        x_c=config.cutoff_c,
        x_h=config.cutoff_h,
        b=config.steepness,
    )
    ln_Pf = config.beta_0 + config.beta_c * Nc + config.beta_h * Nh
    return {"Nc": Nc, "Nh": Nh, "ln_Pf": ln_Pf}
```

### Distance computation

Uses the matmul identity to avoid the `(N_probe, N_env, 3)` broadcast intermediate:

```python
def _pairwise_dist(probe_coords, env_coords):
    # matmul identity: ||a-b||² = ||a||² - 2a·bᵀ + ||b||²
    p_sq  = jnp.sum(probe_coords**2, axis=1, keepdims=True)   # (N_p, 1)
    e_sq  = jnp.sum(env_coords**2,   axis=1, keepdims=True)   # (N_e, 1)
    cross = probe_coords @ env_coords.T                         # (N_p, N_e) GEMM
    dist_sq = jnp.maximum(0.0, p_sq - 2.0*cross + e_sq.T)     # clamp float errors
    return jnp.sqrt(dist_sq + 1e-10)                           # ε prevents ∂/∂0 = ±∞
```

The `+1e-10` epsilon inside `sqrt` (not a `where`-mask) ensures the gradient
`1/(2√(dist_sq+ε))` is finite everywhere. Exact zero distances can only arise from
padding atoms, never from physically distinct probe/environment pairs.

### Switching function

```python
def sigmoid_switch(dist, r0, b):
    """Wan et al. JCTC 2020 canonical form."""
    return jax.nn.sigmoid(b * (r0 - dist))
    # At r=r0: s=0.5; r<<r0: s→1 (contact); r>>r0: s→0 (no contact)
    # Transition half-width ≈ 2·ln(3)/b Å (b=10 → ~0.22 Å, close to hard cutoff)
```

### NaN-safe gradient pattern

JAX evaluates both branches of `jnp.where(cond, f(x), 0.0)` during autodiff.
If `f(x)` produces NaN/Inf for masked elements, the NaN propagates into the backward
pass even though the forward output is masked. Always sanitise the **input** before
the function, not just the output:

```python
# WRONG — NaN propagates through gradient of sqrt(0) even though output is masked
result = jnp.where(mask, jnp.sqrt(x), 0.0)

# CORRECT — safe input fed to function; NaN never produced
safe_x = jnp.where(mask, x, 1.0)          # dummy safe value for masked elements
result = jnp.where(mask, jnp.sqrt(safe_x), 0.0)
```

For distance calculations, the `+1e-10` epsilon eliminates the need for masking
entirely, since `sqrt(ε)` has a finite gradient `1/(2√ε)`.

### Chunked fallback for large systems (>50K atoms)

For systems exceeding the GPU memory budget for the dense approach, chunk over probe
atoms with `jax.checkpoint` to bound backward-pass memory:

```python
@jax.jit
def chunked_contacts(coords, probe_idx, env_idx, excl_mask, r0, b,
                     chunk_size=256, batch_size=4):
    env_coords = coords[env_idx]
    CHUNK = chunk_size

    @jax.checkpoint   # recompute forward during backward → O(CHUNK×N_env) memory
    def process_chunk(_, chunk_data):
        p_idx, mask_chunk = chunk_data
        p_coords = coords[p_idx]
        dist  = _pairwise_dist(p_coords, env_coords)
        counts = jnp.sum(jax.nn.sigmoid(b * (r0 - dist)) * mask_chunk, axis=-1)
        return _, counts

    probe_chunks = probe_idx.reshape(-1, CHUNK)
    mask_chunks  = excl_mask.reshape(-1, CHUNK, excl_mask.shape[-1])
    _, all_counts = jax.lax.scan(process_chunk, None, (probe_chunks, mask_chunks))
    return all_counts.reshape(-1)
```

---

## Trajectory processing

```python
@jax.jit
def mean_protection_factors(
    trajectory: jax.Array,     # (T, N_atoms, 3)
    features:   HDXFeatures,
    config:     HDXConfig,
    weights:    jax.Array | None = None,   # (T,) normalised; None = uniform mean
) -> dict[str, jax.Array]:
    """Ensemble-averaged protection factors."""

    @jax.checkpoint
    def per_frame(coords_t):
        return hdx_forward(coords_t, features, config)["ln_Pf"]

    all_ln_Pf = jax.lax.map(per_frame, trajectory, batch_size=config.batch_size)
    # all_ln_Pf: (T, n_res)

    if weights is None:
        mean_ln_Pf = jnp.mean(all_ln_Pf, axis=0)
    else:
        mean_ln_Pf = jnp.sum(weights[:, None] * all_ln_Pf, axis=0)

    return {"ln_Pf": mean_ln_Pf}
```

`jax.checkpoint` on the per-frame body stores only `(N_atoms, 3)` coordinates per
frame during the backward pass; all intermediate tensors (distance matrices,
contact counts) are recomputed. This trades ~2× forward compute for O(batch_size)
backward memory regardless of trajectory length.

---

## Uptake prediction (`hdx/hdxrate.py`)

```python
@partial(jax.jit, static_argnames=["timepoints"])
def predict_uptake(
    ln_Pf:      jax.Array,         # (n_res,) from hdx_forward
    kint:       jax.Array,         # (n_res,) float32, static — np.nan → 0 before JIT
    can_exchange: jax.Array,       # (n_res,) float32 (0/1)
    peptide_mask: jax.Array,       # (n_peptides, n_res) float32 — peptide coverage
    timepoints:   tuple[float, ...],
) -> jax.Array:                    # (n_peptides, n_timepoints)
    """
    D(t) = Σ_k can_exchange_k * (1 - exp(-kint_k * exp(-ln_Pf_k) * t))
    Summed over residues in each peptide.
    Differentiable through ln_Pf; kint is a static constant.
    """
    # kint has np.nan for non-exchangeable; replace with 0 before passing to JIT
    # can_exchange mask additionally guards against any residual non-zero kint
    def uptake_at_t(t):
        k_eff = kint * jnp.exp(-ln_Pf)                        # (n_res,) effective rate
        d_res = can_exchange * (1.0 - jnp.exp(-k_eff * t))    # (n_res,)
        return peptide_mask @ d_res                             # (n_peptides,)

    return jnp.stack([uptake_at_t(t) for t in timepoints], axis=-1)
```

---

## Wan et al. parameter grid search

The distance matrices are the expensive step (O(N²)); re-evaluating the switching
function across a parameter grid is negligible once distances are cached.

```python
def wan_grid_search(
    dist_c:     np.ndarray,    # (T, n_res, n_heavy) cached per-frame distances
    dist_h:     np.ndarray,    # (T, n_res, n_bb_O)  cached per-frame distances
    excl_mask_c: np.ndarray,   # (n_res, n_heavy)
    excl_mask_h: np.ndarray,   # (n_res, n_bb_O)
    x_c_grid:   jax.Array,     # (K,)  e.g. jnp.arange(5.0, 8.5, 0.5)
    x_h_grid:   jax.Array,     # (M,)  e.g. jnp.arange(2.0, 2.8, 0.1)
    b_grid:     jax.Array,     # (L,)  e.g. jnp.arange(3.0, 21.0, 1.0)
) -> tuple[jax.Array, jax.Array]:
    """
    Returns mean_Nc (K, L, n_res) and mean_Nh (M, L, n_res)
    averaged over T trajectory frames.
    """
    # Vectorise over (x_c, b) grid for Nc
    def nc_for_params(x_c, b):
        # dist_c: (T, n_res, n_heavy)
        contacts = jax.nn.sigmoid(b * (x_c - dist_c))        # (T, n_res, n_heavy)
        return jnp.mean(
            jnp.sum(contacts * excl_mask_c[None], axis=-1),   # mean over T
            axis=0
        )  # (n_res,)

    mean_Nc = jax.vmap(jax.vmap(nc_for_params, (None, 0)), (0, None))(x_c_grid, b_grid)
    # (K, L, n_res)

    # Similarly for Nh with x_h_grid
    def nh_for_params(x_h, b):
        contacts = jax.nn.sigmoid(b * (x_h - dist_h))
        return jnp.mean(jnp.sum(contacts * excl_mask_h[None], axis=-1), axis=0)

    mean_Nh = jax.vmap(jax.vmap(nh_for_params, (None, 0)), (0, None))(x_h_grid, b_grid)
    # (M, L, n_res)

    return mean_Nc, mean_Nh
```

The full Wan et al. grid covers K=7 × L=18 = 126 combinations for Nc and M=8 × L=18
= 144 for Nh. At 50,000 trajectory frames, this sweep runs in seconds on GPU.

---

## JIT shape strategy

JAX recompiles for every distinct input shape. To avoid per-protein recompilation,
pad all arrays to the nearest power-of-2 bucket:

```python
BUCKETS = [512, 1024, 2048, 4096, 8192, 16384,   # single-chain
           32768, 65536, 131072]                   # multi-chain complexes

def get_bucket(n: int) -> int:
    for b in BUCKETS:
        if n <= b: return b
    return BUCKETS[-1]
```

Padding atoms receive `chain_id = "-1"` and `resid = -9999`; the exclusion mask
constructor automatically zeroes them out. At most 9 JIT compilations are cached
per session regardless of how many different proteins are processed.

---

## Memory budget

| System | N_atoms | n_probe | n_heavy | Nc dist matrix | With gradients | 16 GB GPU? |
|---|---|---|---|---|---|---|
| Small (~2,500 atoms) | 2,500 | 250 | 2,500 | 2.5 MB | ~8 MB | Trivial |
| Medium (~5,000 atoms) | 5,000 | 500 | 5,000 | 10 MB | ~30 MB | Trivial |
| Large (~16,000 atoms) | 16,000 | 1,600 | 16,000 | 102 MB | ~400 MB | Comfortable |
| Dimer (~32,000 atoms) | 32,000 | 3,200 | 32,000 | 410 MB | ~1.2 GB | Comfortable |
| Medium complex (~65,000 atoms) | 65,000 | 6,500 | 65,000 | 1.7 GB | ~5 GB | A100/H100 |
| Large complex (>100,000 atoms) | >100,000 | >10,000 | >100,000 | >4 GB | >12 GB | Use chunked |

The asymmetric probe/environment geometry keeps the matrix ~10× smaller than
all-vs-all (n_probe ≈ n_atoms/10 for typical globular proteins).

---

## Public interfaces

### `featurise`

Topology-heavy; uses biotite; runs once per protein topology.

```python
features: HDXFeatures = hdx.featurise(
    structure,           # biotite AtomArray, PDB path, mmCIF path, or .xtc trajectory
    config=HDXConfig(),
    output_index=None,   # None = auto (all protein chains, no HETATM)
)
features.save("my_protein")   # writes my_protein_features.npz + my_protein_topology.json
```

### `forward`

Coordinate-only; no biotite; JIT-compiled; the primary target for HPC batch jobs.

```python
# Load pre-computed features (no biotite needed)
features = HDXFeatures.load("my_protein")

# Single structure
result = hdx.forward(coords, features, config)
# {"Nc": (n_res,), "Nh": (n_res,), "ln_Pf": (n_res,)}

# Trajectory — uniform ensemble mean
result = hdx.forward(coords_traj, features, config)          # coords_traj: (T, N, 3)

# Trajectory — weighted ensemble (BME / MaxEnt)
result = hdx.forward(coords_traj, features, config, weights=w)
```

### `predict`

End-to-end convenience wrapper.

```python
result = hdx.predict("protein.pdb", config=HDXConfig())
result = hdx.predict("protein.xtc", topology="protein.pdb", config=HDXConfig())
```

---

## Output serialisation

```
{prefix}_hdx_output.npz     # keys: Nc, Nh, ln_Pf, [uptake_curves if hdxrate]
{prefix}_hdx_index.json     # res_keys, res_names, can_exchange, kint — alignment table
```

The `_hdx_index.json` provides the mapping from output array indices to
`"chain:resid"` canonical keys for alignment to experimental HDX-MS peptide
coverage maps. All float arrays stored as float32; index arrays as int32.

---

## Validation target

Compare `ln_Pf` against the reference BV model implementation using Wan et al.
default parameters (`x_c=6.5, x_h=2.4, b=10`) on ubiquitin (PDB: 1UBQ).
Expected: Pearson r > 0.95 vs hard-cutoff BV reference values.

Key numerical invariants:

| Test | Expected |
|---|---|
| `sigmoid_switch(r0, r0, b)` | 0.5 for all b |
| `tanh_switch(r0, r0, b/2)` ≈ `sigmoid_switch(r0, r0, b)` | < 1e-6 difference |
| `jax.grad` of scalar loss through `hdx_forward` | no NaN |
| Nc=0, Nh=0 for residue with all-zero contact mask | `ln_Pf = beta_0` |
| Pro/N-term residues | absent from output arrays |
| HDXrate: `kint[0]` for each chain | `np.nan` (N-terminal) |