# BioFeaturisers — Architecture Design

## 1. Overview

`BioFeaturisers` computes differentiable structural biology observables (HDX-MS protection
factors, SAXS profiles) from atomic trajectories or single structures.  The package is
built on three composable interfaces that separate concerns cleanly for HPC workflows:

```
predict    : structures → output data          (end-to-end convenience)
featurise  : structures → input features       (topology-heavy, run once)
forward    : input features → output data      (coordinate-only, run many times)
```

All heavy computation lives in `forward`, which accepts pre-computed feature objects
and raw coordinate arrays, enabling featurisation to run once on a login node and
forward passes to run in parallel on GPU nodes without re-reading structure files.

---

## 2. Package layout

```
BioFeaturisers/
├── core/
│   ├── topology.py          # Biotite-based topology parsing → MinimalTopology
│   ├── contacts.py          # Pure-JAX asymmetric pairwise contact engine
│   ├── output_index.py      # OutputIndex: chain/residue/atom selection logic
│   └── ensemble.py          # Per-structure and weighted-ensemble forward dispatch
│
├── hdx/
│   ├── featurise.py         # HDXFeatures: indices, masks, exclusion arrays
│   ├── forward.py           # BV model JAX kernels (Nc, Nh, ln_Pf)
│   ├── predict.py           # End-to-end: structure(s) → protection factors
│   └── hdxrate.py           # Optional HDXrate integration (kint, uptake curves)
│
├── saxs/
│   ├── featurise.py         # SAXSFeatures: form factors, output_index, q-grid
│   ├── forward.py           # FoXS 6-partial-sum JAX kernels
│   ├── predict.py           # End-to-end: structure(s) → I(q)
│   └── hydration.py         # c1/c2 grid search and analytic gradients
│
├── io/
│   ├── save.py              # Save features + outputs to npz + topology JSON
│   ├── load.py              # Reload features from disk, bypassing biotite
│   └── formats.py           # q-value loaders, experimental data readers
│
├── config.py                # Dataclasses: HDXConfig, SAXSConfig, ContactConfig
├── env.py                   # ComputeConfig: device, memory allocation, backend
└── cli.py                   # Typer-based CLI
```

---

## 3. Core data objects

### 3.1 MinimalTopology

Serialisable JSON object capturing only what downstream modules need.  Produced by
`core/topology.py` from a biotite `AtomArray`.  Saved alongside every npz output so
that results can be aligned to sequences without the original structure file.

```python
@dataclass
class MinimalTopology:
    # Per-atom arrays (length = n_atoms_total, BEFORE any output_index filtering)
    atom_names:    np.ndarray   # (N,) str  — e.g. "N", "CA", "C", "O", "CB"
    res_names:     np.ndarray   # (N,) str  — three-letter residue names
    res_ids:       np.ndarray   # (N,) int  — residue sequence numbers
    chain_ids:     np.ndarray   # (N,) str  — chain identifiers
    element:       np.ndarray   # (N,) str  — element symbols
    is_hetatm:     np.ndarray   # (N,) bool — True for HETATM/ligand atoms
    is_backbone:   np.ndarray   # (N,) bool — N, CA, C, O
    seg_ids:       np.ndarray   # (N,) str  — segment IDs for multi-model files

    # Per-residue arrays (length = n_residues)
    # Built from unique (chain_id, res_id) pairs in sorted order
    res_unique_ids: np.ndarray  # (R,) str  — "A:42", "B:17" etc — canonical keys
    res_can_exchange: np.ndarray  # (R,) bool — False for Pro, N-term, user mask

    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, d: dict) -> "MinimalTopology": ...
    @classmethod
    def from_biotite(cls, atom_array) -> "MinimalTopology": ...
```

**Exchange eligibility** (`res_can_exchange`):
- `False` automatically for: Proline, N-terminal residue of each chain
- Disulfide-bonded Cys: flagged `False` by default, overridable via `HDXConfig`
- User can supply a residue mask via `HDXConfig.exchange_mask` (list of
  `chain:resid` keys) to additionally exclude specific residues

### 3.2 OutputIndex

An `OutputIndex` controls which atoms/residues contribute to and appear in the output
of both HDX and SAXS computations.  It is constructed from a `MinimalTopology` and a
selection specification, then stored in the feature objects.

```python
@dataclass
class OutputIndex:
    """Atom-level boolean mask + derived residue mask.
    Applies to both probe set and environment set selection.
    """
    # Selection applied to the full atom array (n_atoms_total)
    atom_mask:    np.ndarray   # (N,) bool — atoms included in environment
    probe_mask:   np.ndarray   # (N,) bool — atoms that ARE probes (subset of atom_mask)
    output_mask:  np.ndarray   # (R,) bool — residues appearing in output arrays

    # Integer index arrays (JAX-ready, pre-gathered)
    atom_idx:     np.ndarray   # (n_env,)   int32 — environment atom indices
    probe_idx:    np.ndarray   # (n_probe,) int32 — probe atom indices
    output_res_idx: np.ndarray # (n_out,)   int32 — output residue indices

    @classmethod
    def from_selection(
        cls,
        topology: MinimalTopology,
        include_chains: list[str] | None = None,   # None = all chains
        exclude_chains: list[str] | None = None,
        include_hetatm: bool = False,              # ligands in environment
        custom_atom_mask: np.ndarray | None = None # override all of the above
    ) -> "OutputIndex": ...
```

**Key design decisions:**
- SAXS: ligands/HETATM can be included in the scattering sum via `include_hetatm=True`
  (they contribute to I(q)) but they never appear in the output residue list
- HDX: ligands contribute to Nc (heavy atom contacts) if `include_hetatm=True`,
  but are never probes and never appear in the protection factor output
- Cross-chain contacts: included by default in Nc/Nh.  Per-chain-only mode available
  via `HDXConfig.intrachain_only = True`
- Sequence separation exclusion mask is always computed per-chain (cross-chain
  contacts are never excluded by the seq-sep rule)

---

## 4. Shared pairwise contact engine (`core/contacts.py`)

Pure-JAX asymmetric contact engine.  No JAX-MD dependency.  Handles the
probe-vs-environment geometry natively.

```python
@partial(jax.jit, static_argnames=["cutoff", "steepness", "chunk_size"])
def soft_contacts(
    coords:       Float[Array, "N_atoms 3"],   # full coordinate array
    probe_idx:    Int[Array, "n_probe"],       # int32
    env_idx:      Int[Array, "n_env"],         # int32
    excl_mask:    Float[Array, "n_probe n_env"], # float32 — 1=include, 0=exclude
    cutoff:       float,
    steepness:    float,
    chunk_size:   int = 0,     # 0 = auto (dense), >0 = chunked
) -> Float[Array, "n_probe"]:                  # float32 — contact counts
    ...
```

**Auto-dispatch logic (at JIT trace time via `static_argnames`):**
- `chunk_size=0`: uses dense `(n_probe, n_env)` matmul path (recommended for all
  proteins ≤ 50 K atoms)
- `chunk_size>0`: uses `lax.scan` over probe chunks + `jax.checkpoint` on scan body

**Distance kernel:** matmul identity `||a-b||² = ||a||² - 2a·b + ||b||²` with
`jnp.maximum(0, ...)` clamp and `+1e-10` inside sqrt.

**Switching function:** `0.5 * (1 - tanh(steepness * (dist - cutoff)))` — smooth,
always differentiable, no singularities.

**NaN-safe gradients:** `safe_mask` pattern throughout — inputs sanitised before
functions, never just outputs masked.

---

## 5. Ensemble / weighted forward dispatch (`core/ensemble.py`)

The `forward` functions for both HDX and SAXS accept coordinate arrays of shape
either `(N_atoms, 3)` (single structure) or `(T, N_atoms, 3)` (trajectory/ensemble).

```python
def apply_forward(
    forward_fn: Callable,      # JAX-jitted function: coords → output
    coords: Float[Array, "N 3"] | Float[Array, "T N 3"],
    weights: Float[Array, "T"] | None, # float32, normalised — None = uniform mean
    batch_size: int = 8,       # lax.map batch_size for trajectory mode
) -> Float[Array, "out"]:
    """
    Single structure:   returns forward_fn(coords)                → shape (out,)
    Trajectory/uniform: returns mean over frames via lax.map      → shape (out,)
    Trajectory/weighted: returns sum(w_t * forward_fn(coords_t))  → shape (out,)
    """
```

Weights enable reweighting applications (e.g. BME, MaxEnt) where the forward model
is called with ensemble weights derived from experimental data — without recomputing
features.

`jax.checkpoint` is applied on the per-frame function body when processing
trajectories, keeping backward-pass memory O(batch_size × per-frame) regardless of
trajectory length.

---

## 6. HDX module

### 6.1 HDXConfig

```python
@dataclass
class HDXConfig:
    # BV model parameters
    beta_c:          float = 0.35   # weight for heavy-atom contacts (Nc)
    beta_h:          float = 2.0    # weight for H-bond contacts (Nh)
    cutoff_c:        float = 6.5    # Angstrom — Nc cutoff
    cutoff_h:        float = 2.4    # Angstrom — Nh cutoff
    steepness_c:     float = 5.0    # tanh steepness for Nc
    steepness_h:     float = 10.0   # tanh steepness for Nh
    seq_sep_min:     int   = 2      # minimum sequence separation for contact inclusion
    intrachain_only: bool  = False  # if True, cross-chain contacts excluded

    # Topology options
    include_hetatm:  bool  = False  # ligand heavy atoms count toward Nc
    disulfide_exchange: bool = False # treat bonded Cys as exchangeable

    # HDXrate integration
    use_hdxrate:     bool  = False  # if True, compute kint and uptake curves
    hdxrate_pH:      float = 7.0
    hdxrate_temp:    float = 298.15 # Kelvin
    timepoints:      list[float] = field(default_factory=list)  # seconds

    # Compute
    chunk_size:      int   = 0      # 0 = auto dense
    batch_size:      int   = 8      # lax.map batch for trajectories
```

### 6.2 HDXFeatures

The static, coordinate-independent feature object.  Saved to disk once per protein.

```python
@dataclass
class HDXFeatures:
    topology:       MinimalTopology
    output_index:   OutputIndex

    # Probe atom indices (in full atom array)
    amide_N_idx:    np.ndarray   # (n_res_exchangeable,) int32
    amide_H_idx:    np.ndarray   # (n_res_exchangeable,) int32

    # Environment atom indices
    heavy_atom_idx: np.ndarray   # (n_heavy,) int32 — for Nc
    backbone_O_idx: np.ndarray   # (n_bb_O,)  int32 — for Nh

    # Exclusion masks (float32, precomputed, static)
    excl_mask_c:    np.ndarray   # (n_res_exchangeable, n_heavy) float32
    excl_mask_h:    np.ndarray   # (n_res_exchangeable, n_bb_O)  float32

    # Residue metadata (aligned to output_index.output_res_idx)
    res_keys:       np.ndarray   # (n_out,) str — "A:42" canonical keys
    res_names:      np.ndarray   # (n_out,) str — three-letter codes
    can_exchange:   np.ndarray   # (n_out,) bool — False for Pro etc.

    # Optional: HDXrate intrinsic rates (requires use_hdxrate=True)
    # np.nan for non-exchangeable residues (Pro, N-term); aligned to res_keys
    kint:           np.ndarray | None  # (n_out,) float32 — s⁻¹

    def save(self, path: str) -> None:
        # Saves features.npz (float32 arrays) + topology.json
        ...

    @classmethod
    def load(cls, path: str) -> "HDXFeatures": ...
```

**Amide H positions:** When H atoms are absent (common in X-ray structures), amide H
coordinates are computed analytically from N, CA, C positions of adjacent residues
using ideal geometry (N-H bond length 1.01 Å, planar peptide constraint).  This is
done inside `featurise` at the biotite level before any JAX code runs.

### 6.3 HDX forward kernels

```python
# hdx/forward.py

@partial(jax.jit, static_argnames=["cutoff_c", "cutoff_h", "steepness_c",
                                    "steepness_h", "chunk_size"])
def hdx_forward(
    coords:       Float[Array, "N_atoms 3"],
    features:     HDXFeatures,  # carries all index/mask arrays
    config:       HDXConfig,
) -> dict[str, Float[Array, "n_exchangeable"]]:
    """
    Returns:
        Nc:     (n_exchangeable,) heavy-atom contact counts
        Nh:     (n_exchangeable,) H-bond contact counts
        ln_Pf:  (n_exchangeable,) log protection factors
    """
```

Output arrays are always aligned to `features.res_keys` — Prolines and other
non-exchangeable residues are absent (not zero-padded), with alignment keys in
`features.res_keys` for downstream merging with experimental data.

### 6.4 HDXrate integration

When `HDXConfig.use_hdxrate = True`, `hdx/hdxrate.py` computes intrinsic exchange
rates using the HDXrate API:

```python
from hdxrate import k_int_from_sequence
# Signature: k_int_from_sequence(sequence: str, temperature: float, pH: float)
#            → np.ndarray shape (len(sequence),), units s⁻¹
#
# HDXrate handles non-exchangeable positions internally:
#   index 0 (N-terminal residue of the supplied sequence) → 0.0
#   Proline residues                                       → 0.0
#
# Example:
#   k_int_from_sequence('HHHHH', 300, 7.)
#   array([0.00e+00, 2.62e+03, 6.30e+01, 6.30e+01, 9.98e-01])
#           ^ N-term                                  ^ C-term
```

**Chain iteration — critical detail:** HDXrate takes a single linear sequence and
treats index 0 as the N-terminus.  Each chain has its own N-terminus, so
`k_int_from_sequence` must be called once per chain, never on a concatenated
multi-chain sequence.  The implementation in `hdx/hdxrate.py`:

```python
def compute_kint(topology: MinimalTopology, pH: float, temperature: float,
                 config: HDXConfig) -> np.ndarray:
    """Returns kint array aligned to topology.res_unique_ids. np.nan where
    non-exchangeable (Pro, N-term, disulfide Cys if config.disulfide_exchange=False).
    """
    kint_by_key: dict[str, float] = {}

    for chain_id in sorted(set(topology.chain_ids)):
        # Gather residues for this chain in sequence order
        chain_mask = topology.chain_ids == chain_id          # atom-level
        chain_res_keys = [                                   # residue-level, ordered
            k for k in topology.res_unique_ids
            if k.startswith(f"{chain_id}:")
        ]
        chain_res_names = [topology.res_names_by_key[k] for k in chain_res_keys]
        one_letter_seq = "".join(three_to_one(r) for r in chain_res_names)

        # One HDXrate call per chain → N-terminal residue of THIS chain gets 0.0
        rates = k_int_from_sequence(one_letter_seq, temperature, pH)

        for key, rate, res_name in zip(chain_res_keys, rates, chain_res_names):
            is_pro   = (res_name == "PRO")
            is_nterm = (key == chain_res_keys[0])
            is_disulfide = (res_name == "CYS" and key in topology.disulfide_keys
                            and not config.disulfide_exchange)

            if is_pro or is_nterm or is_disulfide:
                kint_by_key[key] = np.nan   # non-exchangeable; rate from HDXrate=0.0
            else:
                kint_by_key[key] = float(rate)

    # Align to topology.res_unique_ids order
    return np.array([kint_by_key[k] for k in topology.res_unique_ids], dtype=np.float32)
```

**Non-exchangeable residue handling summary:**

| Residue type | HDXrate returns | Stored in `kint` | In probe arrays? |
|---|---|---|---|
| N-terminal residue (per chain) | 0.0 | `np.nan` | No |
| Proline | 0.0 | `np.nan` | No |
| Disulfide Cys (default) | rate | `np.nan` | No |
| All other residues | rate > 0 | rate (s⁻¹) | Yes |

`np.nan` is used (not 0.0) to unambiguously flag "not computed" vs. a genuine near-zero
rate, which matters when writing the index JSON for alignment to experimental data.

**Uptake prediction:** the deuterium uptake for a peptide spanning residues i..j is:

```
D(t) = Σ_{k=i}^{j}  can_exchange_k  *  (1 - exp(-kint_k * exp(-ln_Pf_k) * t))
```

The `can_exchange` guard (False where `kint` is `np.nan`) makes the sum robust to
any non-exchangeable residues within the peptide span.  This step is differentiable
through `ln_Pf` and runs in JAX.  `kint` values are constants (not differentiated)
since they are precomputed topology features cached in `HDXFeatures.kint`.

This entire computation is Python-level and runs once during `featurise`; it does
not re-run during `forward` calls.

---

## 7. SAXS module

### 7.1 SAXSConfig

```python
@dataclass
class SAXSConfig:
    # Q-grid
    q_min:         float = 0.01   # Å⁻¹
    q_max:         float = 0.50   # Å⁻¹
    n_q:           int   = 300

    # FoXS model
    c1:            float = 1.0    # excluded volume scaling (1 = no scaling)
    c2:            float = 0.0    # hydration layer amplitude
    fit_c1_c2:     bool  = True   # grid-search c1/c2 on predict
    c1_range:      tuple = (0.95, 1.12)
    c2_range:      tuple = (0.0, 4.0)
    c1_steps:      int   = 18
    c2_steps:      int   = 17
    rho0:          float = 0.334  # solvent electron density (e/Å³)

    # Scattering factor table
    ff_table:      str   = "waasmaier_kirfel"  # or "cromer_mann"

    # Compute
    chunk_size:    int   = 512    # B in chunked Debye sum
    batch_size:    int   = 4      # lax.map batch for trajectories

    # OutputIndex options (same semantics as HDX)
    include_chains:   list[str] | None = None
    exclude_chains:   list[str] | None = None
    include_hetatm:   bool = False
```

### 7.2 SAXSFeatures

```python
@dataclass
class SAXSFeatures:
    topology:       MinimalTopology
    output_index:   OutputIndex    # controls which atoms contribute to I(q)
                                   # and which chains appear in output metadata

    # Atom selection (post output_index filtering)
    atom_idx:       np.ndarray     # (n_atoms_selected,) int32

    # Form factor tables — (n_atoms_selected, Q, 3) for vac/excl/water
    ff_vac:         np.ndarray     # (n_atoms_selected, n_q) float32
    ff_excl:        np.ndarray     # (n_atoms_selected, n_q) float32
    ff_water:       np.ndarray     # (n_atoms_selected, n_q) float32 — rank-1 * Si
    solvent_acc:    np.ndarray     # (n_atoms_selected,) float32 — Si fractions
                                   # Computed once in featurise via biotite SASA
                                   # (Lee-Richards shrake-rupley algorithm).
                                   # Si = SASA_i / max_SASA_i, normalised per element.
                                   # NOTE: static — does not update with coordinates.
                                   # Future: replaced by a differentiable JAX SASA
                                   # kernel so Si gradients flow to coords.

    q_values:       np.ndarray     # (n_q,) float32

    # Metadata
    chain_ids:      np.ndarray     # (n_atoms_selected,) str — for reporting

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "SAXSFeatures": ...
```

**OutputIndex for SAXS:** controls which atoms enter the Debye sum.  Use cases:
- Exclude ligands: `include_hetatm=False` (default)
- Single-chain SAXS from a complex: `include_chains=["A"]`
- Exclude disordered terminal region: `custom_atom_mask`

The output of `saxs_forward` is always a single `I(q)` curve (shape `(n_q,)`); the
`output_index` determines which atoms *contribute*, not which residues *appear* in a
table (unlike HDX).  Chain/selection metadata is recorded in `SAXSFeatures` for
provenance only.

### 7.3 SAXS forward kernels

```python
# saxs/forward.py

@partial(jax.jit, static_argnames=["chunk_size"])
def saxs_six_partials(
    coords:   Float[Array, "N_atoms 3"],      # full array
    features: SAXSFeatures,   # carries atom_idx, ff tables, q_values
    chunk_size: int = 512,
) -> Float[Array, "6 n_q"]:               # [Iaa, Icc, Iss, Iac, Ias, Ics]
    """Chunked double-lax.scan Debye partial sums."""

def saxs_combine(
    partials: Float[Array, "6 n_q"],
    c1: float,
    c2: float,
) -> Float[Array, "n_q"]:               # I(q)
    """Polynomial recombination — differentiable w.r.t. c1, c2."""

def saxs_forward(
    coords:   Float[Array, "N_atoms 3"],
    features: SAXSFeatures,
    c1:       float = 1.0,
    c2:       float = 0.0,
) -> Float[Array, "n_q"]:               # I(q)
```

**Key implementation notes:**
- Form factors are gathered per selected atom: `ff_vac[atom_idx]` etc., outside JIT
- The double-`where` / custom VJP pattern for `sinc(qr)` at `r=0`
- Diagonal (i=j) contribution computed separately as `Σ_i F_i(q)²` to avoid
  the `sinc(0)` singularity entirely
- `donate_argnums=(0,)` on the JIT boundary to allow XLA to recycle coord buffer

---

## 8. Input / Output (`io/`)

### 8.1 Feature serialisation

`HDXFeatures.save(path)` and `SAXSFeatures.save(path)` each write two files:

```
{prefix}_features.npz      # all float32/int32 arrays
{prefix}_topology.json     # MinimalTopology as JSON
```

The `.npz` contains array keys matching the dataclass field names.  Loading is
symmetric: `HDXFeatures.load(prefix)` reconstructs the full object without biotite.

### 8.2 Output serialisation

Forward pass outputs are saved as:

```
{prefix}_hdx_output.npz    # keys: Nc, Nh, ln_Pf, [uptake_curves if hdxrate]
{prefix}_hdx_index.json    # res_keys, res_names, can_exchange — alignment table
{prefix}_saxs_output.npz   # keys: I_q, q_values, [partials if requested]
{prefix}_saxs_index.json   # chain_ids, atom_counts, c1, c2 used
```

All float arrays stored as **float32**.  Integer index arrays stored as **int32**.

The `_index.json` files provide the minimal metadata needed to align output arrays
to experimental data (e.g. matching `res_keys` to HDX-MS peptide coverage maps).

### 8.3 Experimental data readers (`io/formats.py`)

- SAXS: `.dat` (3-column q/I/sigma), ATSAS `.fit`, generic CSV
- HDX: `.csv` in standard HDX-MS format (peptide, start, end, timepoint, deuterium)
- q-value arrays: from file or auto-generated from `SAXSConfig`

---

## 9. Three public interfaces

### 9.1 `featurise`

Topology-heavy.  Uses biotite.  Runs once per protein topology.

```python
# HDX
features: HDXFeatures = hdx.featurise(
    structure,          # biotite AtomArray, PDB path, mmCIF path, or trajectory
    config=HDXConfig(),
    output_index=None,  # None = auto (all protein chains, no HETATM)
)
features.save("my_protein")

# SAXS
features: SAXSFeatures = saxs.featurise(
    structure,
    config=SAXSConfig(),
    output_index=None,
)
features.save("my_protein")
```

### 9.2 `forward`

Coordinate-only.  No biotite.  JIT-compiled.  The primary target for HPC batch jobs.

```python
# HDX — single structure
result = hdx.forward(coords, features, config)
# result: {"Nc": ..., "Nh": ..., "ln_Pf": ...}

# HDX — trajectory (uniform mean)
result = hdx.forward(coords_traj, features, config)

# HDX — trajectory with weights
result = hdx.forward(coords_traj, features, config, weights=w)

# SAXS — single structure
I_q = saxs.forward(coords, features, config)

# SAXS — trajectory (uniform mean)
I_q = saxs.forward(coords_traj, features, config)

# SAXS — trajectory with weights  
I_q = saxs.forward(coords_traj, features, config, weights=w)
```

### 9.3 `predict`

End-to-end convenience.  Calls `featurise` then `forward`.

```python
result = hdx.predict("protein.pdb", config=HDXConfig())
I_q    = saxs.predict("protein.pdb", config=SAXSConfig())

# Multi-frame (trajectory)
result = hdx.predict("protein.xtc", topology="protein.pdb", config=HDXConfig())
```

---

## 10. Compute environment (`env.py`)

`ComputeConfig` controls JAX initialisation and device placement.  It must be
instantiated (and `configure()` called) **before any JAX import that triggers
compilation**, typically at the top of a script or at CLI startup.

```python
from enum import Enum

class Backend(str, Enum):
    JAX   = "jax"
    TORCH = "torch"   # deferred — not yet implemented

class Device(str, Enum):
    CPU  = "cpu"
    GPU  = "gpu"      # uses first available CUDA/ROCm device
    AUTO = "auto"     # GPU if available, else CPU

@dataclass
class ComputeConfig:
    backend:           Backend = Backend.JAX

    # Device selection
    device:            Device  = Device.AUTO
    gpu_index:         int     = 0         # which GPU when multiple are present

    # JAX memory behaviour
    # False (default): JAX preallocates 75% of GPU VRAM on first use.
    # True: disables preallocation — JAX allocates as needed.
    # Recommended True on shared HPC nodes; False for throughput-critical jobs.
    disable_preallocation: bool = False

    # JAX precision
    enable_x64:        bool    = False     # float64 support (slower on consumer GPUs)

    def configure(self) -> None:
        """Apply settings.  Call once before any JAX computation."""
        if self.backend == Backend.JAX:
            self._configure_jax()

    def _configure_jax(self) -> None:
        import os
        # Must be set before jax is imported to take effect
        if self.device == Device.CPU:
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        elif self.device == Device.GPU:
            os.environ.setdefault("JAX_PLATFORMS", "cuda")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu_index))
        # AUTO: let JAX pick; CUDA if available, else CPU

        import jax
        if self.disable_preallocation:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            # Alternative fine-grained cap:
            # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
        if self.enable_x64:
            jax.config.update("jax_enable_x64", True)
```

**Usage patterns:**

```python
# Default — GPU with preallocation (best throughput)
cfg = ComputeConfig()
cfg.configure()

# Shared HPC node — disable preallocation, force CPU
cfg = ComputeConfig(device=Device.CPU, disable_preallocation=True)
cfg.configure()

# Multi-GPU node — use second GPU
cfg = ComputeConfig(device=Device.GPU, gpu_index=1)
cfg.configure()
```

**CLI integration:** `--env` accepts a TOML file or inline key=value pairs that
map to `ComputeConfig` fields, applied before any computation:

```
biofeaturisers hdx predict --structure protein.pdb \
    --env device=gpu,gpu_index=1,disable_preallocation=true
```

**Future (Torch backend):** when `backend=Backend.TORCH` is activated, `configure()`
will call `torch.device(...)` and set default tensor dtype.  The `ComputeConfig`
dataclass will grow `torch_dtype` and `torch_compile` fields at that point.  The
`forward` modules will dispatch to JAX or Torch kernels based on
`ComputeConfig.backend` at call time; feature objects remain backend-agnostic NumPy
throughout.

---

## 11. CLI (`cli.py`)

Built with Typer.  Subcommand structure: `biofeaturisers <method> <interface>`.

```
biofeaturisers hdx featurise  --structure protein.pdb  --config hdx.toml  --out protein
biofeaturisers hdx forward    --features protein        --coords traj.npz  --out results
biofeaturisers hdx predict    --structure protein.pdb   --config hdx.toml  --out results

biofeaturisers saxs featurise --structure protein.pdb  --config saxs.toml --out protein
biofeaturisers saxs forward   --features protein        --coords traj.npz  --out results
biofeaturisers saxs predict   --structure protein.pdb   --config saxs.toml --out results
```

Config files are TOML, mapping directly to `HDXConfig` / `SAXSConfig` dataclass
fields.  Any config field can also be passed as `--key value` on the command line,
overriding the TOML value.  `ComputeConfig` is always applied first via `--env`
(see Section 10).

---

## 12. Key design decisions (rationale)

| Decision | Choice | Rationale |
|---|---|---|
| Topology library | biotite | Handles multi-chain, HETATM, mmCIF natively; no OpenMM/MDTraj dependency |
| Contact engine | Pure JAX dense matrix | Asymmetric probe/env geometry; proteins ≤50K atoms fit in GPU VRAM; no JAX-MD asymmetric query support |
| Switching function | tanh | No singularities, always differentiable, `k` controls width |
| Pairwise distances | matmul identity | Avoids (N,N,3) intermediate; leverages GEMM |
| Trajectory batching | `lax.map(batch_size=B)` | Bounded memory; differentiable; JIT-compatible |
| Gradient memory | `jax.checkpoint` on scan body | O(batch_size) memory cost for backward pass through trajectories |
| JIT shape strategy | Power-of-2 atom buckets | ≤6 compilations per session; static shapes for XLA |
| Serialisation | npz + JSON | Inspectable, language-agnostic, float32 |
| HDXrate | Optional dependency, per-chain iteration | Not always available; chain N-termini correctly zeroed per chain |
| SAXS SASA | Biotite Lee-Richards | Confirmed; differential SASA kernel deferred to future milestone |
| SAXS model | FoXS 6-partial-sum | c1/c2 fitting outside O(N²) loop; differentiable w.r.t. coords and c1/c2 |
| Cross-chain contacts | Included by default | Physically correct for complexes; intrachain_only flag available |
| OutputIndex | Atom-level mask on both HDX and SAXS | Consistent API; handles complexes, ligands, partial chain SAXS |
| Amide H generation | Analytic from N/CA/C if absent | Handles X-ray structures without H; placed in featurise, not forward |
| Compute config | `ComputeConfig` in `env.py` | JAX preallocation + device selection before compilation; Torch backend hook ready |

---

## 13. Open questions / deferred decisions

- **HDXrate API**: confirmed — `k_int_from_sequence(sequence, temperature, pH)`;
  per-chain iteration pattern fully documented in §6.4.
- **Solvent-accessible surface (Si)**: confirmed — biotite Shrake-Rupley in
  `featurise`; differential JAX SASA kernel deferred to a future milestone (see
  `SAXSFeatures.solvent_acc` note in §7.2).
- **Torch backend**: deferred — implement and validate JAX first, then port `forward`
  modules.  Feature objects are backend-agnostic NumPy throughout; `ComputeConfig`
  already has the dispatch hook.  Will gain `torch_dtype` and `torch_compile` fields.
- **CryoEM**: architecture accommodates a future `cryoem/` submodule following the
  same featurise/forward/predict pattern (see CryoEM context document).