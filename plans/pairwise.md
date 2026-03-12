# BioFeaturisers — Shared Pairwise Components Design

## Overview

The three experimental modules (HDX, SAXS, CryoEM) all require pairwise distance
computation between atomic positions. They diverge precisely at the point where they
begin using those distances differently:

- **HDX** — soft scalar contact count per probe residue (sigmoid/tanh switching)
- **SAXS** — sinc-weighted sum over a Q-grid (Debye formula)
- **CryoEM** — Gaussian-weighted sum over a 3D voxel grid

The shared layer provides everything up to and including safe distance primitives,
switching functions, and trajectory dispatch. The boundary rule is strict: anything
that takes **only** `(coords, indices)` lives in `core/`. As soon as form factors,
sinc, contact masks, or voxel grids enter, it belongs to the module-specific layer.

```
core/
├── pairwise.py      ← distance kernels (asymmetric and symmetric block)
├── safe_math.py     ← double-where primitives: safe_sqrt, safe_sinc, safe_mask
├── switching.py     ← soft step functions + Wan et al. grid-search utility
└── ensemble.py      ← trajectory dispatch: single / uniform mean / weighted mean
```

---

## Module consumption map

| Module  | Distance fn                            | Safe math                     | Switching          | Adds on top                                   |
|---------|----------------------------------------|-------------------------------|--------------------|-----------------------------------------------|
| HDX     | `dist_matrix_asymmetric`               | `safe_sqrt` via ε in matmul   | `sigmoid_switch`   | excl_mask multiply → `sum(axis=-1)` → ln_Pf   |
| SAXS    | `dist_matrix_block` + `dist_from_sq_block` | `safe_sinc` (custom VJP)  | —                  | `qr = q * dist`, six-partial einsum           |
| CryoEM  | `dist_matrix_asymmetric` (atom→voxel)  | `safe_sqrt`                   | —                  | `exp(-dist²/2σ²)`, voxel accumulation         |

---

## `core/pairwise.py`

### Design rationale: two distinct distance geometries

**Asymmetric** `(n_probe, n_env)` is used by HDX (amide N/H → heavy atoms) and
CryoEM (atoms → voxel grid). The matmul identity `||a-b||² = ||a||² - 2·a·bᵀ + ||b||²`
avoids the `(N_p, N_e, 3)` broadcast intermediate and leverages GPU GEMM.

**Symmetric block** `(B, B)` is used by SAXS inside each chunk of the Debye sum.
Broadcasting (`diff = c_i[:,None,:] - c_j[None,:,:]`) is preferred here over the
matmul identity because atoms within the same molecule have similar coordinate
magnitudes, making `||a||² - 2a·b + ||b||²` susceptible to catastrophic cancellation
at small separations.

### The `dist_sq` vs `dist` interface decision

The symmetric block path deliberately returns `dist_sq` (not `dist`) from
`dist_matrix_block`. This is because SAXS must handle the diagonal singularity
before forming `qr = q * dist`. The caller chooses one of two strategies:

1. Mask out the diagonal and handle `Σ F_i(q)²` analytically via
   `diagonal_self_pairs` — slightly faster.
2. Pass `dist_sq` through `dist_from_sq_block` which applies the double-where
   pattern, setting `dist=0` on the diagonal. This makes `qr=0` and `sinc(0)=1`,
   recovering the correct diagonal automatically.

Both are correct. The `dist_sq` interface supports both without forcing a choice.

For the asymmetric path (HDX, CryoEM), the `+1e-10` inside sqrt is sufficient
because probe and environment atoms are distinct sets — exact zero distances do not
occur in practice.

```python
# core/pairwise.py

from __future__ import annotations
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
import chex

# ── Asymmetric: (n_probe, n_env) ─────────────────────────────────────────────

def dist_matrix_asymmetric(probe_coords: Float[Array, "N_p 3"],
                            env_coords:   Float[Array, "N_e 3"]) -> Float[Array, "N_p N_e"]:
    """
    ||p_i - e_j||  for all (i, j) pairs.

    Uses the matmul identity  ||a-b||² = ||a||² - 2·a·bᵀ + ||b||²
    to avoid the (N_p, N_e, 3) intermediate.  Peak memory: O(N_p × N_e).

    Args:
        probe_coords : (N_p, 3)
        env_coords   : (N_e, 3)

    Returns:
        dist : (N_p, N_e)  float32, always ≥ 0, no NaN, no Inf.
    """
    p_sq  = jnp.sum(probe_coords ** 2, axis=1, keepdims=True)  # (N_p, 1)
    e_sq  = jnp.sum(env_coords   ** 2, axis=1, keepdims=True)  # (N_e, 1)
    cross = probe_coords @ env_coords.T                         # (N_p, N_e) GEMM
    dist_sq = jnp.maximum(0.0, p_sq - 2.0 * cross + e_sq.T)   # clamp float errors
    return jnp.sqrt(dist_sq + 1e-10)                           # ε prevents ∂/∂0 = ±∞


# ── Symmetric block: (B, B) ───────────────────────────────────────────────────

def dist_matrix_block(coords_i: Float[Array, "B_i 3"],
                       coords_j: Float[Array, "B_j 3"]) -> Float[Array, "B_i B_j"]:
    """
    Returns dist_sq (not dist) for a (B_i, B_j) block.

    Explicit broadcasting avoids catastrophic cancellation at small separations.
    Returns raw squared distances — may contain exact zeros on the i==j diagonal
    when called with the same block.  Caller applies dist_from_sq_block or
    diagonal_self_pairs before proceeding.

    Args:
        coords_i : (B_i, 3)
        coords_j : (B_j, 3)

    Returns:
        dist_sq : (B_i, B_j)
    """
    diff = coords_i[:, None, :] - coords_j[None, :, :]  # (B_i, B_j, 3)
    return jnp.sum(diff ** 2, axis=-1)                   # (B_i, B_j)


def dist_from_sq_block(dist_sq: Float[Array, "B B"]) -> Float[Array, "B B"]:
    """
    Safe sqrt over a (B, B) block that may contain zeros on the diagonal.
    Uses the double-where pattern so gradients are well-defined at r=0.

    Returns:
        dist : (B, B)  — zero where dist_sq == 0, √dist_sq elsewhere.
    """
    is_zero = dist_sq <= 0.0
    safe_sq = jnp.where(is_zero, 1.0, dist_sq)  # dummy input in safe branch
    return jnp.where(is_zero, 0.0, jnp.sqrt(safe_sq))


# ── Chunked asymmetric fallback (>50K-atom systems) ──────────────────────────

@partial(jax.jit, static_argnames=["chunk_size"])
def chunked_dist_apply(
    probe_coords : Float[Array, "N_p 3"],
    env_coords   : Float[Array, "N_e 3"],
    apply_fn,
    chunk_size   : int = 256,
) -> Float[Array, "N_p out_dim"]:
    """
    Apply `apply_fn(chunk_of_probes, env_coords)` over probe chunks and
    concatenate results.  Keeps peak memory at O(chunk_size × N_e).

    Probes are padded to a multiple of chunk_size with zeros; padding
    rows are discarded from the output before returning.

    Returns:
        (N_p, out_dim)
    """
    N_p  = probe_coords.shape[0]
    pad  = (-N_p) % chunk_size
    p_pad = jnp.pad(probe_coords, ((0, pad), (0, 0)))

    n_chunks = p_pad.shape[0] // chunk_size
    chunks   = p_pad.reshape(n_chunks, chunk_size, 3)

    @jax.checkpoint
    def body(_, chunk):
        return None, apply_fn(chunk, env_coords)  # (C, out_dim)

    _, results = jax.lax.scan(body, None, chunks)
    return results.reshape(-1, results.shape[-1])[:N_p]
```

---

## `core/safe_math.py`

Every JAX pitfall with `where` + singular functions is isolated here.
**The invariant: always sanitise the input, never just the output.**

JAX evaluates both branches of `jnp.where(cond, f(x), 0.0)` during autodiff.
If `f(x)` produces NaN or Inf for masked elements (e.g. `sqrt(0)` has infinite
gradient), that NaN propagates into the backward pass even though the forward
output is correctly masked. The fix is to replace masked inputs with a safe
dummy value *before* calling `f`, then mask the output.

```python
# core/safe_math.py

from __future__ import annotations
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def safe_sqrt(x: Float[Array, "..."], eps: float = 1e-10) -> Float[Array, "..."]:
    """
    sqrt(x + eps).
    The eps inside the sqrt (not a where-mask) ensures the gradient
    1/(2√(x+ε)) is finite everywhere.  Used on the asymmetric HDX/CryoEM
    path where probe ≠ env atoms, so exact-zero distances are only possible
    from padding — not from physically meaningful pairs.
    """
    return jnp.sqrt(x + eps)


def safe_sqrt_sym(dist_sq: Float[Array, "B B"]) -> Float[Array, "B B"]:
    """
    Double-where sqrt for symmetric blocks that may have exact zeros on the
    diagonal (i == j self-pairs).

    Gradient is zero at the origin — correct because self-pair distances
    do not generate forces — rather than the infinite gradient from sqrt(0).

    Strict double-where:  safe input  →  safe function  →  masked output.
    """
    is_zero = dist_sq <= 0.0
    safe_sq = jnp.where(is_zero, 1.0, dist_sq)
    return jnp.where(is_zero, 0.0, jnp.sqrt(safe_sq))


# ── sinc kernel (most numerically delicate primitive) ────────────────────────

@jax.custom_vjp
def safe_sinc(qr: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    sinc(qr) = sin(qr)/qr  with Taylor expansion at the origin.

    Forward pass:
      - qr > 1e-8  →  sin(qr)/qr               (normal branch)
      - qr ≤ 1e-8  →  1 - qr²/6 + qr⁴/120     (Taylor branch)

    The custom VJP avoids accumulating an autodiff graph through sin/division.
    Instead it stores (qr, sinc(qr)) as residuals and uses the analytic
    derivative:
        sinc'(x) = (cos(x) - sinc(x)) / x
    At x→0, Taylor gives  -x/3 + x³/30  (gradient is 0 at origin — correct).
    """
    safe_qr = jnp.where(qr > 1e-8, qr, 1.0)
    return jnp.where(
        qr > 1e-8,
        jnp.sin(safe_qr) / safe_qr,
        1.0 - qr ** 2 / 6.0 + qr ** 4 / 120.0,
    )


def _safe_sinc_fwd(qr):
    y = safe_sinc(qr)
    return y, (qr, y)  # store both to avoid recomputing sinc in backward


def _safe_sinc_bwd(res, g):
    qr, y    = res
    safe_qr  = jnp.where(qr > 1e-8, qr, 1.0)
    dsinc = jnp.where(
        qr > 1e-8,
        (jnp.cos(safe_qr) - y) / safe_qr,
        -qr / 3.0 + qr ** 3 / 30.0,
    )
    return (g * dsinc,)


safe_sinc.defvjp(_safe_sinc_fwd, _safe_sinc_bwd)


# ── Generic safe-mask ─────────────────────────────────────────────────────────

def safe_mask(
    mask        : Float[Array, "..."] | Array, # Boolean or float mask
    fn,
    operand     : Float[Array, "..."],
    placeholder : float = 0.0,
    safe_val    : float = 0.5,
) -> Float[Array, "..."]:
    """
    Apply fn(operand) only where mask is True, with NaN-safe gradients.

    The safe_val substitution happens BEFORE calling fn, preventing NaN/Inf
    gradients from propagating through the False branch.

    Args:
        mask        : boolean array, same shape as operand
        fn          : element-wise function that may produce NaN/Inf at 0
        operand     : input array
        placeholder : value written to output where mask is False
        safe_val    : dummy input substituted for masked elements
    """
    safe_operand = jnp.where(mask, operand, safe_val)
    return jnp.where(mask, fn(safe_operand), placeholder)


# ── SAXS diagonal contribution ────────────────────────────────────────────────

def diagonal_self_pairs(ff: Float[Array, "N Q"]) -> Float[Array, "Q"]:
    """
    Compute the i == j diagonal contribution to the Debye sum analytically:
        Σ_i  F_i(q)²

    sinc(0) = 1 always, so the diagonal reduces to the sum of squared form
    factors.  Handling this outside the chunked loop avoids the sinc(0)
    singularity entirely.  Gradients: ∂(F²)/∂F = 2F — trivially clean.

    Args:
        ff : (N, Q)  effective form factors
    Returns:
        (Q,)
    """
    return jnp.sum(ff ** 2, axis=0)
```

---

## `core/switching.py`

### The Wan et al. JCTC 2020 sigmoid switch (canonical HDX form)

The Best-Vendruscolo contact counts are defined as:

$$N_c = \sum_j \frac{\exp(-b(r_j - x_c))}{1 + \exp(-b(r_j - x_c))}, \quad N_h = \sum_j \frac{\exp(-b(r_j - x_h))}{1 + \exp(-b(r_j - x_h))}$$

which is identically `jax.nn.sigmoid(b * (x_c - r_j))`.  Wan et al. validated this
form against 1 ms of ubiquitin (72 amides, 50,000 frames at 300 K) and 12.5 µs of
BPTI (30 amides, 50,000 frames).

Their systematic parameter exploration covers:

| Parameter | Range    | Step | Notes                               |
|-----------|----------|------|-------------------------------------|
| `b`       | 3–20 Å⁻¹ | 1    | sharpness of the switching function |
| `x_c`     | 5.0–8.0 Å | 0.5 | heavy-atom contact cutoff           |
| `x_h`     | 2.0–2.7 Å | 0.1 | H-bond contact cutoff               |

**Relation to tanh:**  `sigmoid(b·Δ) = 0.5·(1 + tanh(b·Δ/2))`
so `sigmoid_switch(b)` is exactly equivalent to `tanh_switch(k = b/2)`.
Use `sigmoid_switch` when comparing against published Wan et al. parameter grids
since their reported `b` values map directly; use `tanh_switch` when `k` is more
natural for the workflow.

### The grid-search utility

The key optimisation for Wan et al.–style sweeps: compute distance matrices once
per frame (the expensive O(N²) step), cache them, then re-apply the switching
function across the full `(x_c, x_h, b)` grid at negligible cost.  For 50,000
trajectory frames × 18 × 17 grid points = 306 combinations, the switching
re-evaluation costs microseconds per frame — all within a single JIT-compiled
`vmap` call.

```python
# core/switching.py

from __future__ import annotations
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


def sigmoid_switch(dist: Float[Array, "..."], r0: float, b: float) -> Float[Array, "..."]:
    """
    Wan et al. JCTC 2020 canonical form:
        s(r) = sigmoid(b · (r₀ − r))   ≡   jax.nn.sigmoid(b * (r0 - dist))

    Properties:
      - At r = r₀:  s = 0.5
      - At r ≪ r₀:  s → 1   (within cutoff — counts as contact)
      - At r ≫ r₀:  s → 0   (outside cutoff — no contact)
      - Transition half-width ≈ 2·ln(3)/b Å
        (b = 10 Å⁻¹ → ~0.22 Å half-width, closely approximating a hard cutoff)

    Unconditionally numerically stable.  Use this form when comparing to
    published Wan et al. BV parameter grids.

    Relation to tanh_switch:  sigmoid_switch(b) ≡ tanh_switch(k = b/2)
    """
    return jax.nn.sigmoid(b * (r0 - dist))


def tanh_switch(dist: Float[Array, "..."], r0: float, k: float) -> Float[Array, "..."]:
    """
    s(r) = 0.5 · (1 − tanh(k · (r − r₀)))

    Equivalent to sigmoid_switch with b = 2k.  Prefer sigmoid_switch for
    published parameter comparison; use tanh_switch when k is more natural
    (e.g. k ≈ 5 Å⁻¹ ↔ b ≈ 10 Å⁻¹).
    """
    return 0.5 * (1.0 - jnp.tanh(k * (dist - r0)))


def rational_switch(dist: Float[Array, "..."], r0: float,
                    n: int = 6, m: int = 12) -> Float[Array, "..."]:
    """
    PLUMED-style rational switching function:
        s(r) = (1 − (r/r₀)ⁿ) / (1 − (r/r₀)ᵐ)

    Has a removable singularity at r = r₀ where the L'Hôpital limit is n/m.
    Double-where on INPUT keeps gradients finite at the singularity.
    Less numerically robust than sigmoid/tanh.  Provided for PLUMED compatibility.
    """
    x      = dist / r0
    near_one = jnp.abs(x - 1.0) < 1e-6
    x_safe = jnp.where(near_one, 0.5, x)      # safe input, far from singularity
    s      = (1.0 - x_safe ** n) / (1.0 - x_safe ** m)
    return jnp.where(near_one, float(n) / float(m), s)


# ── Wan et al. grid search over (r0, b) ──────────────────────────────────────

def apply_switch_grid(
    dist_matrix : Float[Array, "N_p N_e"],   # cached per frame
    excl_mask   : Float[Array, "N_p N_e"],   # float32 validity mask
    r0_grid     : Float[Array, "K"],         # cutoff values to sweep
    b_grid      : Float[Array, "L"],         # sharpness values to sweep
) -> Float[Array, "K L N_p"]:
    """
    Vectorise the sigmoid switch over a (K, L) hyperparameter grid.

    Distances are computed once per frame (the expensive step); only the
    switching function evaluation is repeated across grid points.

    Workflow:
        1. For each trajectory frame, call dist_matrix_asymmetric once
           to produce dist_c (N_p, N_heavy) and dist_h (N_p, N_bb_O).
        2. Optionally cache these matrices on CPU (e.g. HDF5 per frame).
        3. Call apply_switch_grid(dist_c, excl_mask_c, x_c_grid, b_grid)
           and apply_switch_grid(dist_h, excl_mask_h, x_h_grid, b_grid)
           to evaluate all (K, L) combinations in microseconds per frame.
        4. Average across frames to get mean Nc and Nh for each grid point.
        5. Compute ln_Pf = beta_c * Nc + beta_h * Nh and compare to experiment.

    Args:
        dist_matrix : (N_p, N_e)  pre-computed distance matrix for one frame
        excl_mask   : (N_p, N_e)  sequence-exclusion + chain-boundary mask
        r0_grid     : (K,)        cutoff values (x_c or x_h)
        b_grid      : (L,)        sharpness values

    Returns:
        contact_grid : (K, L, N_p)  contact counts per (cutoff, sharpness, residue)
    """
    def for_r0(r0):
        def for_b(b):
            contacts = jax.nn.sigmoid(b * (r0 - dist_matrix))  # (N_p, N_e)
            return jnp.sum(contacts * excl_mask, axis=-1)        # (N_p,)
        return jax.vmap(for_b)(b_grid)                          # (L, N_p)

    return jax.vmap(for_r0)(r0_grid)                            # (K, L, N_p)


# ── Full BV contact calculation (Wan et al. JCTC 2020) ───────────────────────

def bv_contact_counts(
    coords       : Float[Array, "N_atoms 3"],
    amide_N_idx  : Int[Array, "N_res"],
    amide_H_idx  : Int[Array, "N_res"],
    heavy_idx    : Int[Array, "N_heavy"],
    backbone_O_idx: Int[Array, "N_bb_O"],
    excl_mask_c  : Float[Array, "N_res N_heavy"],
    excl_mask_h  : Float[Array, "N_res N_bb_O"],
    x_c          : float = 6.5,
    x_h          : float = 2.4,
    b            : float = 10.0,
) -> tuple[Float[Array, "N_res"], Float[Array, "N_res"]]:
    """
    Compute Best-Vendruscolo contact counts Nc and Nh using the Wan et al.
    JCTC 2020 sigmoid switching function.

    Nc : heavy-atom contacts around each amide N
         N_c^i = Σ_j  sigmoid(b · (x_c − ||r_N^i − r_j||)) · mask_c[i,j]

    Nh : backbone H-bond contacts around each amide H
         N_h^i = Σ_j  sigmoid(b · (x_h − ||r_H^i − r_O^j||)) · mask_h[i,j]

    The exclusion mask (excl_mask_c / excl_mask_h) zeroes out:
      - Self-residue contacts (seq_sep = 0)
      - Nearest-neighbour contacts (seq_sep ≤ 2, configurable)
      - Cross-chain contacts (always allowed; mask handles chain boundaries)
      - Padding atoms (chain_id == -1 sentinel)

    Protection factor (computed downstream, not here):
        ln_Pf^i = beta_c · Nc^i + beta_h · Nh^i
        (beta_c = 0.35, beta_h = 2.0  — Lindorff-Larsen / Best-Vendruscolo defaults)

    Args:
        coords        : full coordinate array for the system
        amide_N_idx   : residue-ordered amide N indices into coords
        amide_H_idx   : residue-ordered amide H indices into coords
                        (analytically placed if absent from structure file)
        heavy_idx     : all heavy atom indices for Nc environment
        backbone_O_idx: backbone O indices for Nh environment
        excl_mask_c   : pre-built float32 validity mask for Nc
        excl_mask_h   : pre-built float32 validity mask for Nh
        x_c           : heavy-atom contact cutoff (default 6.5 Å)
        x_h           : H-bond contact cutoff (default 2.4 Å)
        b             : sigmoid sharpness (default 10.0 Å⁻¹)

    Returns:
        Nc : (N_res,)  heavy-atom contact counts
        Nh : (N_res,)  H-bond contact counts
    """
    from .pairwise import dist_matrix_asymmetric  # local import to avoid circularity

    # N_c : amide N → all heavy atoms
    N_coords     = coords[amide_N_idx]   # (N_res, 3)
    heavy_coords = coords[heavy_idx]     # (N_heavy, 3)
    dist_c = dist_matrix_asymmetric(N_coords, heavy_coords)   # (N_res, N_heavy)
    Nc = jnp.sum(jax.nn.sigmoid(b * (x_c - dist_c)) * excl_mask_c, axis=-1)

    # N_h : amide H → backbone O
    H_coords = coords[amide_H_idx]       # (N_res, 3)
    O_coords = coords[backbone_O_idx]    # (N_bb_O, 3)
    dist_h = dist_matrix_asymmetric(H_coords, O_coords)       # (N_res, N_bb_O)
    Nh = jnp.sum(jax.nn.sigmoid(b * (x_h - dist_h)) * excl_mask_h, axis=-1)

    return Nc, Nh
```

---

## `core/ensemble.py`

```python
# core/ensemble.py

from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def apply_forward(
    forward_fn  : Callable[[Float[Array, "N 3"]], Float[Array, "out"]],
    coords      : Float[Array, "N 3"] | Float[Array, "T N 3"],
    weights     : Float[Array, "T"] | None,   # normalised weights, None = uniform mean
    batch_size  : int = 8,
) -> Float[Array, "out"]:
    """
    Dispatch `forward_fn` over a single structure or a trajectory/ensemble.

    Single structure (N, 3):
        returns forward_fn(coords)                    → shape (out,)

    Trajectory, uniform mean (T, N, 3), weights=None:
        returns mean(forward_fn(coords_t))            → shape (out,)
        lax.map(batch_size) + jax.checkpoint for bounded memory.

    Trajectory, weighted mean (T, N, 3), weights=(T,):
        returns Σ_t w_t · forward_fn(coords_t)       → shape (out,)
        Same memory strategy; enables BME / MaxEnt reweighting.

    Memory cost  : batch_size × per_frame_memory (forward only).
    Backward cost: O(batch_size) activations stored; remainder recomputed
                   via jax.checkpoint on the per-frame body.
    """
    if coords.ndim == 2:
        return forward_fn(coords)  # single structure — no batching

    @jax.checkpoint
    def _per_frame(coords_t: jax.Array) -> jax.Array:
        return forward_fn(coords_t)

    all_outputs = jax.lax.map(_per_frame, coords, batch_size=batch_size)  # (T, out)

    if weights is None:
        return jnp.mean(all_outputs, axis=0)
    else:
        return jnp.sum(weights[:, None] * all_outputs, axis=0)


def weighted_ensemble(
    forward_fn : Callable[[Float[Array, "N 3"]], Float[Array, "out"]],
    coords     : Float[Array, "T N 3"],
    weights    : Float[Array, "T"],   # must sum to 1
    batch_size : int = 8,
) -> Float[Array, "out"]:
    """Convenience wrapper: weighted ensemble average."""
    return apply_forward(forward_fn, coords, weights, batch_size)


def uniform_ensemble(
    forward_fn : Callable[[Float[Array, "N 3"]], Float[Array, "out"]],
    coords     : Float[Array, "T N 3"],
    batch_size : int = 8,
) -> Float[Array, "out"]:
    """Convenience wrapper: uniform ensemble average (trajectory mean)."""
    return apply_forward(forward_fn, coords, None, batch_size)
```

---

## Implementation plan

### Step 1 — Core shared primitives

**Files to write:** `core/pairwise.py`, `core/safe_math.py`, `core/switching.py`,
`core/ensemble.py`

**Key implementation tasks:**
- `dist_matrix_asymmetric` — matmul identity, `jnp.maximum(0, ...)` clamp, `+1e-10` in sqrt
- `dist_matrix_block` + `dist_from_sq_block` — broadcast diff, double-where on `dist_sq`
- `diagonal_self_pairs` — `jnp.sum(ff**2, axis=0)`, used by SAXS to skip the diagonal entirely
- `safe_sqrt` and `safe_sqrt_sym` — ε-inside vs double-where for the two distance geometries
- `safe_sinc` with custom VJP — the most critical primitive for correctness; store `(qr, y)` as residuals
- `safe_mask` — generic pattern: `jnp.where(mask, operand, safe_val)` before fn, then mask output
- `sigmoid_switch` and `tanh_switch` — both with explicit equivalence comment (`b = 2k`)
- `bv_contact_counts` — full Nc / Nh computation per Wan et al. JCTC 2020
- `apply_switch_grid` — distance-cached grid sweep for Wan et al. parameter exploration
- `apply_forward` — single/trajectory dispatch with `lax.map(batch_size)` + `jax.checkpoint`
- `chunked_dist_apply` — fallback for >50K-atom systems via scan over probe chunks

**Validation targets for this step:**
- `safe_sinc(0.0)` → 1.0, `jax.grad(safe_sinc)(0.0)` → 0.0 (not NaN)
- `dist_from_sq_block` at zero → 0.0 with finite gradient
- `sigmoid_switch(r0, r0, b)` → 0.5 for any b
- `tanh_switch(r0, r0, b/2)` matches `sigmoid_switch(r0, r0, b)` to float32 tolerance

### Step 2 — HDX module

**Files to write:** `hdx/featurise.py`, `hdx/forward.py`, `hdx/hdxrate.py`,
`hdx/predict.py`

**Key implementation tasks:**
- `HDXFeatures` dataclass — indices, exclusion masks, topology metadata, optional `kint`
- `build_exclusion_mask` — chain-aware seq-sep exclusion: cross-chain contacts always allowed,
  `same_chain & |resid_i - resid_j| <= min_sep` → 0.0
- Amide H placement — analytic from N/CA/C if H atoms absent (X-ray structures)
- `hdx_forward` — calls `bv_contact_counts` from `core/switching.py`, returns `Nc`, `Nh`, `ln_Pf`
- `compute_kint` — per-chain HDXrate calls (`k_int_from_sequence(seq, T, pH)`), one call
  per chain so each chain N-terminus is correctly zeroed; `np.nan` for non-exchangeable
  residues (Pro, N-term, disulfide Cys); never concatenate multi-chain sequences
- Uptake prediction — `D(t) = Σ_k can_exchange_k * (1 - exp(-kint_k * exp(-ln_Pf_k) * t))`
  differentiable through `ln_Pf`; `kint` is a static constant

**Validation target:** compare `ln_Pf` against the reference BV model implementation
using the Wan et al. default parameters (`x_c=6.5, x_h=2.4, b=10`) on a test PDB
(e.g. ubiquitin 1UBQ).  Expected: Pearson r > 0.95 vs hard-cutoff BV values.

### Step 3 — SAXS module

**Files to write:** `saxs/form_factors.py`, `saxs/debye.py`, `saxs/foxs.py`,
`saxs/predict.py`

**Key implementation tasks:**
- Waasmaier-Kirfel 5-Gaussian tables — `(T, 5)` coefficients; `ff_table[atom_types]` gather
- Excluded volume form factors — Fraser Gaussian sphere model; `f_excl(q) = ρ₀·V·exp(...)`
- Water form factor — `f_water(q) = 2·f_H + f_O − f_excl_water`; rank-1 in atom dimension
  after multiplication by `solvent_acc` (SASA fractions, static from `featurise`)
- Six-partial-sum block — single sinc kernel, three weighted vectors `w_v/w_e/w_s` via einsum,
  six bilinear contractions; `diagonal_self_pairs` handles the i==j contribution separately
- `saxs_combine` — polynomial recombination `Iaa - c1·Iac + c1²·Icc + c2·Ias - c1c2·Ics + c2²·Iss`;
  fitting c1/c2 requires no recomputation of the O(N²) Debye sums
- c1/c2 grid search — analytic scale factor; vectorised over precomputed partials

**Validation target:** compare `I(q)` against CRYSOL / FoXS output on the same PDB.
Expected: χ² < 1.1 on a noise-free theoretical profile comparison.

### Step 4 — CryoEM module (future)

**Files to write:** `cryoem/featurise.py`, `cryoem/forward.py`, `cryoem/predict.py`

Uses `dist_matrix_asymmetric` (atom → voxel grid points) and `safe_sqrt`.  Adds
Gaussian-weighted voxel accumulation on top of the shared primitives.  Architecture
accommodated in the current design; deferred until HDX and SAXS are validated.

### Step 5 — JAX optimisation

- Profile per-module forward pass with `jax.profiler.trace`
- Tune `chunk_size` and `batch_size` for target GPU (A100: B=1024, Q=300 comfortable;
  consumer 8 GB: B=256–512)
- Verify `donate_argnums=(0,)` on the JIT boundary for coordinate buffer recycling
- Benchmark `lax.scan` vs `lax.map(batch_size=B)` for trajectory throughput

### Step 6 — PyTorch port

All feature objects (`HDXFeatures`, `SAXSFeatures`) are backend-agnostic NumPy.
The `forward` modules dispatch on `ComputeConfig.backend`.  Port the JAX kernels
to PyTorch, validate outputs match JAX to float32 tolerance, then benchmark.

---

## Key numerical invariants (test checklist)

| Test | Expected |
|------|----------|
| `safe_sinc(0.0)` | 1.0 |
| `jax.grad(safe_sinc)(0.0)` | 0.0 (not NaN) |
| `dist_from_sq_block(jnp.zeros((4,4)))` | `jnp.zeros((4,4))` |
| `jax.grad(lambda x: dist_from_sq_block(x).sum())(jnp.zeros((4,4)))` | finite (not NaN) |
| `sigmoid_switch(r0, r0, b)` | 0.5 for all b |
| `tanh_switch(r0, r0, b/2)` ≈ `sigmoid_switch(r0, r0, b)` | < 1e-6 difference |
| `bv_contact_counts` with zero coords and identity mask | all zeros (no contacts) |
| `apply_forward` on trajectory, `jax.grad` of scalar loss | no NaN in grad |
| HDX: `ln_Pf` on 1UBQ vs hard-cutoff BV | Pearson r > 0.95 |
| SAXS: `I(q)` vs CRYSOL on test PDB | χ² < 1.1 |