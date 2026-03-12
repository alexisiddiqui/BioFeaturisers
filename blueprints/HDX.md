# Differentiable spatial contact counting in JAX for HDX protection factors

**The dense pairwise distance matrix is the recommended primary approach for the Best-Vendruscolo HDX model across all practical protein sizes.** For the asymmetric probe-vs-environment geometry of this problem (~N_residues probes querying ~N_atoms environment), the dense matrix fits comfortably in GPU memory even for 16,000-atom proteins, is trivially JIT-compiled, fully differentiable, and avoids all the complexity of neighbor-list overflow management. A smooth `tanh` switching function replaces the hard cutoff to enable `jax.grad`, and `lax.map` with `batch_size` handles trajectory frames with bounded memory. This report provides complete implementation patterns, memory analysis, and concrete code for every component.

---

## 1. Dense O(N²) distance matrix: the workhorse approach

The dense approach computes all pairwise distances between probe atoms and environment atoms in a single matrix operation. For the BV model, this means an `(N_probe, N_env)` matrix where `N_probe` is the number of backbone amide atoms and `N_env` is the number of heavy atoms (for N_c) or backbone oxygens (for N_h).

**Three implementation patterns exist in JAX**, each with different memory characteristics. The matmul identity `‖a−b‖² = ‖a‖² − 2a·b + ‖b‖²` is the most memory-efficient because it avoids the `(N_probe, N_env, 3)` intermediate tensor from broadcasting:

```python
def pairwise_dist_matmul(probe_coords, env_coords):
    """Memory-efficient: only creates (N_probe, N_env) intermediates."""
    p_sq = jnp.sum(probe_coords**2, axis=1, keepdims=True)   # (N_p, 1)
    e_sq = jnp.sum(env_coords**2, axis=1, keepdims=True)     # (N_e, 1)
    cross = probe_coords @ env_coords.T                       # (N_p, N_e) via GEMM
    dist_sq = jnp.maximum(0.0, p_sq - 2*cross + e_sq.T)      # clamp float errors
    return jnp.sqrt(dist_sq + 1e-10)
```

The broadcasting approach `diff = P[:, None, :] - E[None, :, :]` creates a 3× larger intermediate but XLA may fuse the subtraction-square-sum chain. The double-vmap approach `vmap(vmap(dist_fn, (None, 0)), (0, None))` is JAX-idiomatic and produces equivalent XLA code after tracing. **For production, use the matmul formulation** — it leverages optimized GPU GEMM kernels and has deterministic peak memory.

### Memory budget for real proteins

| System | N_atoms (total) | N_probe | N_env | Distance matrix | With gradients (~3×) | 16GB GPU? |
|--------|-----------------|---------|-------|-----------------|---------------------|-----------|
| Small single-chain (~2,500 atoms) | 2,500 | 250 | 2,500 | **2.5 MB** | ~8 MB | Trivial |
| Medium single-chain (~5,000 atoms) | 5,000 | 500 | 5,000 | **10 MB** | ~30 MB | Trivial |
| Large single-chain (~16,000 atoms) | 16,000 | 1,600 | 16,000 | **102 MB** | ~400 MB | Comfortable |
| Small complex / dimer (~32,000 atoms) | 32,000 | 3,200 | 32,000 | **410 MB** | ~1.2 GB | Comfortable |
| Medium complex (~65,000 atoms) | 65,000 | 6,500 | 65,000 | **1.7 GB** | ~5 GB | Fits (A100/H100) |
| Large complex / virus capsid (~100,000 atoms) | 100,000 | 10,000 | 100,000 | **4.0 GB** | ~12 GB | Tight on 16GB — chunk |
| Very large complex (>100,000 atoms) | >100,000 | >10,000 | >100,000 | >4 GB | >12 GB | **Use chunked fallback** |

JAX preallocates **75% of GPU VRAM** by default (~12 GB on a 16 GB card). The asymmetric geometry of the BV model — where `N_probe ≈ N_residues ≈ N_atoms/10` — keeps the matrix roughly 10× smaller than all-vs-all. For multi-chain complexes, `N_atoms` is the **total concatenated atom count across all chains**, so a homodimer is simply treated as a single 32,000-atom system. The dense approach remains viable up to ~65,000 atoms on an A100 (80 GB); beyond that, the chunked fallback (§10) keeps peak memory at O(CHUNK × N_env) regardless of complex size.

### Chunking for larger systems

When memory is tight, chunk over probe atoms using `jax.lax.map` with `batch_size`:

```python
@jax.jit
def chunked_contacts(coords, probe_idx, env_idx, excl_mask, cutoff, steepness):
    env_coords = coords[env_idx]
    def per_probe_chunk(start_idx):
        chunk_probes = jax.lax.dynamic_slice(probe_idx, (start_idx,), (CHUNK,))
        chunk_mask = jax.lax.dynamic_slice(excl_mask, (start_idx, 0), (CHUNK, excl_mask.shape[1]))
        p_coords = coords[chunk_probes]
        dist = pairwise_dist_matmul(p_coords, env_coords)
        switch = 0.5 * (1.0 - jnp.tanh(steepness * (dist - cutoff)))
        return jnp.sum(switch * chunk_mask, axis=-1)
    starts = jnp.arange(0, probe_idx.shape[0], CHUNK)
    return lax.map(per_probe_chunk, starts, batch_size=4)
```

With `batch_size=4` and `CHUNK=256`, peak memory drops to `4 × 256 × N_env × 4 bytes` — about **16 MB for N_env=16,000**. Combine with `jax.checkpoint` on the scan body to prevent gradient memory from scaling with the number of chunks.

---

## 2. JAX-MD neighbor lists: architecture and limitations

JAX-MD (`jax_md.partition.neighbor_list`) is the most mature spatial partitioning library in the JAX ecosystem, designed by Schoenholz & Cubuk (NeurIPS 2020) specifically for differentiable physics under XLA's static-shape constraint.

### The allocate/update split

The core design resolves XLA's static-shape requirement through a two-phase API:

- **`neighbor_fn.allocate(R)`** — Python-level, inspects positions to determine `max_occupancy`, creates fixed-size arrays. **Cannot be JIT-compiled** because it determines shapes dynamically.
- **`neighbor_fn.update(R, neighbors)`** — JIT-compatible, fills the pre-allocated arrays with new neighbor data. If capacity is exceeded, sets `did_buffer_overflow = True` but continues with truncated results.

The canonical simulation loop checks overflow outside JIT and reallocates when needed:

```python
nbrs = neighbor_fn.allocate(R)
for epoch in range(n_epochs):
    new_state, nbrs = lax.fori_loop(0, 100, body_fn, (state, nbrs))
    if nbrs.did_buffer_overflow:
        nbrs = neighbor_fn.allocate(state.position)  # reallocate outside JIT
    else:
        state = new_state
```

### Three neighbor list formats

**Dense** `(N, max_neighbors_per_atom)` stores per-particle neighbor lists padded with sentinel index `N`. Best when neighbor counts are uniform. **Sparse** `(2, total_neighbors)` stores sender/receiver edge pairs — more memory-efficient when connectivity varies. **OrderedSparse** stores only `i < j` pairs for ~2× speedup on symmetric potentials; converts to `jraph.GraphsTuple` via `partition.to_jraph()`.

### Cell list construction with static shapes

Under the hood, `neighbor_list` uses a cell list for O(N log N) construction. The cell list uses a clever scatter-based parallel assignment: particles are sorted by cell hash (`hash = sum(floor(pos/cell_size) * stride_multipliers)`), each assigned a unique slot `cell_id = hash * cell_capacity + local_index`, then copied into a flat buffer via a single `lax.scatter` call. Each cell has a fixed `cell_capacity` — overflow triggers a flag just like the neighbor list.

### Critical limitation for the BV model: no asymmetric queries

**JAX-MD's `neighbor_list` only supports a single particle set.** The API takes one position array `R` of shape `[N, dim]` and finds neighbors among those same N particles. There is no built-in support for asymmetric queries where a probe set queries an environment set. To use JAX-MD for the BV model, you would need to concatenate probe and environment atoms into one array and use `custom_mask_function` to filter unwanted same-set interactions — an awkward workaround that wastes compute on probe-probe and env-env pairs.

**This is why the dense approach is preferable for the BV model**: the probe-vs-environment geometry is naturally asymmetric, and the matrix sizes are small enough that O(N²) is not a bottleneck.

---

## 3. Switching functions for differentiability

The original BV model uses hard Heaviside cutoffs, which have zero gradient everywhere and are incompatible with `jax.grad`. Three smooth replacements exist, ordered by practical preference:

### Tanh switching (recommended)

```python
def tanh_switch(r, r0, k=5.0):
    """s(r) = 0.5 * (1 - tanh(k*(r - r0)))
    At r=r0: s=0.5. Transition width ≈ 2/k Å."""
    return 0.5 * (1.0 - jnp.tanh(k * (r - r0)))
```

Always numerically stable, always differentiable, no singularities. With `k=5.0`, the transition is ~0.4 Å wide — narrow enough to closely approximate a hard cutoff while providing smooth gradients. The `jnp.tanh` implementation in JAX is numerically robust for all inputs.

### Rational switching (PLUMED-style)

The function `s(r) = (1 − (r/r₀)ⁿ) / (1 − (r/r₀)ᵐ)` with n=6, m=12 has a singularity at r = r₀ where both numerator and denominator approach zero. The L'Hôpital limit is `n/m = 0.5`. Safe implementation requires the **double-where pattern**:

```python
def rational_switch(r, r0, n=6, m=12):
    x = r / r0
    near_one = jnp.abs(x - 1.0) < 1e-6
    x_safe = jnp.where(near_one, 0.5, x)  # safe input, never near singularity
    s = (1.0 - x_safe**n) / (1.0 - x_safe**m)
    return jnp.where(near_one, n/m, s)     # substitute limit value at singularity
```

### Sigmoid switching (Wan et al. JCTC 2020 — validated reference form)

This is the **canonical published form** from Wan et al. (*J. Chem. Theory Comput.* 2020), validated against 1-ms ubiquitin (72 amides, 50,000 frames at 300 K) and a 12.5-μs segment of native-state BPTI (30 amides, 50,000 frames). They write it explicitly as:

$$N_c = \sum_j \frac{\exp(-b(x_j - x_c))}{1 + \exp(-b(x_j - x_c))}, \quad N_h = \sum_j \frac{\exp(-b(x_j - x_h))}{1 + \exp(-b(x_j - x_h))}$$

which is identically `jax.nn.sigmoid(b * (x_c - x_j))`. Always smooth, no special handling needed, never exactly 0 or 1. The **relationship to the tanh switch** is exact: `sigmoid(b*(r0 - r)) = 0.5*(1 + tanh(b*(r0 - r)/2))`, so the Wan sigmoid with sharpness `b` is equivalent to the tanh switch with `k = b/2`. Both are numerically correct; **use the sigmoid form when comparing against published BV parameter grids** since `b` maps directly to Wan et al. tables.

The Wan model extends the standard BV equation with an **additional intercept term β₀**:

$$\ln(\text{PF}_i) = \beta_c \cdot N_c^i + \beta_h \cdot N_h^i + \beta_0$$

β₀ compensates for correlations between the heavy-atom contact and hydrogen-bond terms — when a residue forms a hydrogen bond it necessarily also has heavy-atom contacts, so the two terms are not independent. β₀ absorbs this systematic co-variance and should be treated as a third fitted parameter alongside β_c and β_h. In the standard BV model β₀ is implicitly zero.

```python
def sigmoid_switch(r, r0, b=10.0):
    """Wan et al. JCTC 2020 form. Equivalent to tanh_switch with k=b/2.
    b=10 Å⁻¹ ≈ tanh steepness k=5 — good starting point."""
    return jax.nn.sigmoid(b * (r0 - r))

# Wan protection factor (β0 is the cooperativity intercept)
ln_Pf = beta_c * Nc + beta_h * Nh + beta_0
```

The Wan et al. paper systematically explores `b ∈ [3, 20] Å⁻¹` (step 1), `x_c ∈ [5.0, 8.0] Å` (step 0.5), and `x_h ∈ [2.0, 2.7] Å` (step 0.1) — treating all three as hyperparameters rather than fixed values. Crucially, the Wan model also includes an **additional cooperativity offset term β₀** absent from the original BV equation:

$$\ln(PF_i) = \beta_0 + \beta_c \cdot N_c^i + \beta_h \cdot N_h^i$$

β₀ compensates for correlations between the heavy-atom contact count and the hydrogen-bond count — when a residue is buried (high N_c) it tends to also be hydrogen-bonded (high N_h), so the two terms are not independent. The offset absorbs this systematic co-variation and improves fit to experimental data. β₀ is an additional fitted parameter alongside β_c and β_h. This grid-search protocol is cheap if raw per-frame distances are cached: the distance matrix is computed once per frame (the expensive step), and the switching function re-evaluation across any `(x_c, x_h, b)` combination costs only μs per frame as a vectorised post-processing step over pre-stored distance arrays.

### The NaN gradient problem and the double-where fix

This is the single most important JAX pitfall for masked molecular computations. When writing `jnp.where(mask, f(x), 0.0)`, **JAX evaluates `f(x)` for ALL elements** — including masked ones. If `f(x)` produces NaN for masked elements (e.g., `sqrt(0)` has infinite gradient, `1/0` produces Inf), the NaN **propagates into the backward pass** even though it's masked in the forward pass.

The fix is JAX-MD's `safe_mask` pattern — sanitize inputs *before* the function:

```python
def safe_mask(mask, fn, operand, placeholder=0.0):
    """Apply fn only where mask is True, with safe gradients."""
    safe_input = jnp.where(mask, operand, 0.5)     # safe value for masked elements
    return jnp.where(mask, fn(safe_input), placeholder)
```

For distance computations specifically, always add epsilon inside `sqrt`: `dist = jnp.sqrt(dist_sq + 1e-10)`. This prevents the infinite gradient at zero distance from corrupting backpropagation through padded neighbor entries.

---

## 4. Sparse/padded neighbor representation

When using neighbor lists (rather than dense matrices), the standard JAX-compatible format is a `(N_probe, max_k)` integer array of neighbor indices, padded with a sentinel value (typically `N_atoms` or the atom's own index) for unused slots.

### The pattern

```python
# neighbor_idx: (N_probe, max_k) — padded with N_atoms sentinel
# mask: (N_probe, max_k) boolean
mask = neighbor_idx < n_atoms

# Safe gather: sanitize before indexing
safe_idx = jnp.where(mask, neighbor_idx, 0)  # 0 is a valid index, won't cause OOB
neighbor_coords = all_coords[safe_idx]        # (N_probe, max_k, 3)

# Compute distances
diff = probe_coords[:, None, :] - neighbor_coords  # (N_probe, max_k, 3)
dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)

# Apply switching and mask
contacts = tanh_switch(dist, cutoff) * mask.astype(jnp.float32)
counts = jnp.sum(contacts, axis=-1)  # (N_probe,)
```

Gradients flow correctly through this pattern because: (1) the gather `all_coords[safe_idx]` has a well-defined backward pass (scatter-add), (2) `jnp.where` with the mask zeros out contributions from padding, and (3) the safe index ensures no out-of-bounds access that could produce garbage gradients. JAX clamps out-of-bounds gather indices by default (`mode='clip'`), but the explicit safe_idx pattern is more transparent and avoids unexpected gradient accumulation on the clamped element.

### Choosing max_k

For typical globular proteins based on heavy atom packing density (~0.055 atoms/ų interior):

| Cutoff | Sphere volume | Expected neighbors | Recommended max_k |
|--------|--------------|-------------------|-------------------|
| **6.5 Å** | 1,150 ų | 25–65 (surface–interior) | **128** |
| **2.4 Å** | 58 ų | 0–3 (strong H-bonds only) | **8** |

A 2.4 Å cutoff captures only very short hydrogen bonds — a backbone amide typically forms 0–1 such contacts. **max_k = 8 is extremely safe** for N_h. For the 6.5 Å cutoff, interior atoms in densely packed regions can have ~60–65 heavy atom neighbors after sequence exclusion; **max_k = 128 provides ~2× headroom**.

---

## 5. Hash grid / voxel approach in pure JAX

A spatial hash grid divides space into cells of size ≥ cutoff, then for each query atom only checks the 3³ = 27 neighboring cells. JAX-MD implements this as the internal acceleration structure for `neighbor_list`.

The key challenge in JAX is that cell occupancy varies — some cells may contain many atoms while others are empty — but XLA requires fixed-size arrays. JAX-MD solves this by allocating `cell_capacity` slots per cell (estimated during the non-JIT `allocate` phase) and using the sort-then-scatter trick:

1. Compute cell hash for each atom: `hash = sum(floor(pos / cell_size) * stride)`
2. Sort atoms by hash (brings same-cell atoms together)
3. Assign slot: `cell_slot = hash * cell_capacity + intra_cell_index`
4. Scatter into flat buffer: `buffer.at[cell_slot].set(atom_data)`
5. Unflatten to grid: reshape to `(n_cells_x, n_cells_y, n_cells_z, cell_capacity, ...)`

**For a pure JAX implementation without JAX-MD**, you could use `jax.ops.segment_sum` to count atoms per cell and build per-cell atom lists. However, this is complex to implement correctly with static shapes and gradient support. For the BV model's problem sizes (up to ~16,000 atoms), the dense approach is simpler and fast enough — the hash grid becomes worthwhile only beyond ~50,000 atoms or in tight inner loops of MD simulations.

**Practical verdict**: implementing a hash grid from scratch in JAX is possible but not worth it for this application. Use JAX-MD's cell list if you need one, or stick with the dense approach.

---

## 6. Trajectory batching strategy

The BV model typically averages protection factors over MD trajectory frames. Three JAX patterns handle frame batching with different memory-compute tradeoffs:

**`jax.vmap` over frames** — materializes all frames simultaneously. Memory scales linearly with frame count. For 100 frames × 400 MB/frame = 40 GB — too large for most GPUs. Best only for very small batch sizes (2–4 frames).

**`jax.lax.scan` with carry** — processes frames sequentially, accumulating a running sum. Memory is constant (one frame). Ideal when you only need the mean contact counts:

```python
def body(running_sum, frame_coords):
    counts = compute_contacts(frame_coords, ...)
    return running_sum + counts, None

mean_counts, _ = lax.scan(body, jnp.zeros(n_probe), trajectory)
mean_counts /= n_frames
```

**`jax.lax.map` with `batch_size`** (recommended) — the hybrid approach. Processes `batch_size` frames in parallel via vmap, iterates over batches sequentially. Memory = `batch_size × per_frame_memory`. With `batch_size=8` and 400 MB/frame, peak memory is ~3.2 GB — fits on a 16 GB GPU:

```python
per_frame_counts = lax.map(
    lambda coords: compute_contacts(coords, ...),
    trajectory,       # (n_frames, n_atoms, 3)
    batch_size=8      # process 8 frames at a time
)  # returns (n_frames, n_probe)
```

**When differentiating through `lax.scan`**, JAX stores residuals for every iteration. Apply `jax.checkpoint` (gradient rematerialization) on the scan body to trade ~2× forward compute for O(1) gradient memory — critical for long trajectories.

### Static shapes across frames

All frames in a trajectory have the same atom count, so the coordinates tensor `(n_frames, n_atoms, 3)` has a single static shape. The probe indices, environment indices, and exclusion masks are identical across frames (topology doesn't change). This means the JIT-compiled function works for all frames without recompilation — a major advantage over approaches that require per-frame neighbor list construction.

---

## 7. JAX-MD as a dependency: practical considerations

**License**: Apache 2.0 (Google Research), permissive for commercial and academic use.

**API for the BV model**: JAX-MD's `neighbor_list` takes a single position array and a `displacement_or_metric` function (from `jax_md.space`). The standard construction:

```python
from jax_md import space, partition
displacement_fn, shift_fn = space.free()  # non-periodic system
neighbor_fn = partition.neighbor_list(
    displacement_fn,
    box=100.0,           # bounding box (must contain all atoms)
    r_cutoff=6.5,
    format=partition.NeighborListFormat.Dense,
    capacity_multiplier=1.25
)
nbrs = neighbor_fn.allocate(positions)
nbrs = nbrs.update(new_positions)
```

**The asymmetric query gap**: JAX-MD does not support querying probe positions against a separate environment set. The workaround — concatenating both sets and masking — works but wastes compute on unwanted probe-probe pairs. For the BV model where `N_probe ≪ N_env`, this overhead is modest (the probe-probe block is ~1% of the total matrix for a 10:1 ratio). Alternatively, use `custom_mask_function` to exclude these pairs.

**Dependency weight**: JAX-MD pulls in JAX, jax-lib, numpy, and optionally jaxlib-cuda. It's a pure Python package with no compiled extensions beyond JAX itself. The partition module can be imported independently without the full simulation stack.

---

## 8. The two-set problem and reusing computations

The BV model computes two contact types from overlapping but distinct geometric queries:

- **N_c**: backbone amide N positions → all heavy atoms within x_c Å
- **N_h**: backbone amide H positions → backbone O atoms within x_h Å

### Efficient implementation

Since amide N and H positions are within ~1 Å of each other (along the N-H bond), and the two environment sets overlap (backbone O ⊂ heavy atoms), there are opportunities to share computation. However, the different cutoffs and different environment sets make full reuse impractical. **The recommended approach is two independent forward passes**, each computing a `(N_probe, N_env)` matrix:

```python
# N_c: amide N → all heavy atoms
Nc = compute_soft_contacts(coords, amide_N_idx, heavy_atom_idx, excl_mask_c, 
                           r0=x_c, b=b)

# N_h: amide H → backbone O
Nh = compute_soft_contacts(coords, amide_H_idx, backbone_O_idx, excl_mask_h,
                           r0=x_h, b=b)

# Protection factor (Wan et al.: add cooperativity offset β₀)
ln_Pf = beta_0 + beta_c * Nc + beta_h * Nh  # β_0 fitted, β_c=0.35, β_h=2.0
```

The N_h computation is tiny: `N_probe × N_backbone_O ≈ N_res × N_res`. For a 1,000-residue protein, this is 1M entries = 4 MB — negligible. The N_c computation dominates at `N_res × N_heavy ≈ 1,000 × 10,000` = 10M entries = 40 MB — still very manageable.

**Both terms share the same `coords` array**, so `jax.grad` of `ln_Pf` with respect to `coords` correctly accumulates gradients from both terms via automatic differentiation's chain rule.

---

## 9. Reference implementations worth studying

**JAX-MD `partition.py`** (github.com/jax-md/jax-md) is the gold standard for static-shape spatial partitioning in JAX. Key functions: `neighbor_list`, `cell_list`, `_neighboring_cells`, and the `safe_mask` utility. The allocate/update/overflow pattern is directly transferable to any JAX spatial computation.

**e3nn-jax** (github.com/e3nn/e3nn-jax) demonstrates the downstream consumer pattern: it expects edge lists `(edge_src, edge_dst)` as input and provides `scatter_sum` for aggregation. It does not construct neighbor lists itself — a useful architectural separation.

**MACE-JAX** (github.com/ACEsuit/mace-jax) shows production-grade fixed-shape batching for molecular GNNs: batches are padded to fixed `n_edge` with derived node/graph padding caps. The JAX-MD integration for running MD simulations demonstrates the full pipeline from neighbor lists to energy/force computation.

**Jraph** (Google DeepMind, now archived → JraphX) provides `GraphsTuple` with padding utilities for JIT compatibility. Its strategy of padding to the nearest power of two for nodes and edges is directly applicable to neighbor list sizing.

**Notable absence**: there is no direct JAX equivalent of `torch_cluster.radius_graph`. PyTorch's `torch_cluster.radius(x, y, r)` natively supports the asymmetric two-set query that JAX-MD lacks. The closest JAX approach for large-scale asymmetric queries would be to adapt JAX-MD's cell list or implement the dense matrix approach described here.

---

## 10. Complete recommended implementation

Given the constraints — all protein sizes up to ~16,000 atoms, CPU + GPU, full autodiff, JIT with static shapes — the following architecture is recommended:

### Primary approach: dense matrix with sigmoid switching

```python
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['b', 'x_c', 'x_h'])
def compute_bv_protection_factors(
    coords,              # (N_atoms, 3)
    amide_N_idx,         # (N_res,) indices of amide N atoms
    amide_H_idx,         # (N_res,) indices of amide H atoms  
    heavy_atom_idx,      # (N_heavy,) indices of all heavy atoms
    backbone_O_idx,      # (N_bb_O,) indices of backbone O atoms
    excl_mask_c,         # (N_res, N_heavy) float32: 1 where contact is valid
    excl_mask_h,         # (N_res, N_bb_O) float32: 1 where contact is valid
    beta_c=0.35,
    beta_h=2.0,
    beta_0=0.0,          # Wan et al. cooperativity intercept; 0.0 = standard BV
    x_c=6.5,             # heavy-atom cutoff (Å); Wan et al. explores 5.0–8.0
    x_h=2.4,             # H-bond cutoff (Å); Wan et al. explores 2.0–2.7
    b=10.0,              # sharpness (Å⁻¹); Wan et al. explores 3–20
):
    # --- N_c: heavy atom contacts around amide N ---
    N_coords = coords[amide_N_idx]                            # (N_res, 3)
    heavy_coords = coords[heavy_atom_idx]                      # (N_heavy, 3)
    
    p_sq = jnp.sum(N_coords**2, axis=1, keepdims=True)
    e_sq = jnp.sum(heavy_coords**2, axis=1, keepdims=True)
    dist_c = jnp.sqrt(
        jnp.maximum(0.0, p_sq - 2*N_coords @ heavy_coords.T + e_sq.T) + 1e-10
    )
    Nc = jnp.sum(
        jax.nn.sigmoid(b * (x_c - dist_c)) * excl_mask_c,
        axis=-1
    )
    
    # --- N_h: backbone O contacts around amide H ---
    H_coords = coords[amide_H_idx]                            # (N_res, 3)
    O_coords = coords[backbone_O_idx]                          # (N_bb_O, 3)
    
    h_sq = jnp.sum(H_coords**2, axis=1, keepdims=True)
    o_sq = jnp.sum(O_coords**2, axis=1, keepdims=True)
    dist_h = jnp.sqrt(
        jnp.maximum(0.0, h_sq - 2*H_coords @ O_coords.T + o_sq.T) + 1e-10
    )
    Nh = jnp.sum(
        jax.nn.sigmoid(b * (x_h - dist_h)) * excl_mask_h,
        axis=-1
    )
    
    # --- Protection factor (Wan et al. form with cooperativity intercept) ---
    ln_Pf = beta_c * Nc + beta_h * Nh + beta_0
    return ln_Pf, Nc, Nh

# Gradients come for free:
grad_fn = jax.grad(lambda c, *a, **k: jnp.sum(compute_bv_protection_factors(c, *a, **k)[0]))
```

### Static shape strategy with multi-chain support

Pad all arrays to fixed maximum sizes (or power-of-2 buckets) and use boolean masks to exclude padding atoms. This avoids JIT recompilation across different protein sizes. **For multi-chain systems**, the bucket covers the total atom count across all chains concatenated into a single coordinate array — no per-chain splitting is needed at the JAX level.

```python
BUCKETS = [512, 1024, 2048, 4096, 8192, 16384,  # single-chain
           32768, 65536, 131072]                  # multi-chain complexes

def pad_to_bucket(arr, bucket_size, pad_value=0.0):
    pad_width = bucket_size - arr.shape[0]
    return jnp.pad(arr, [(0, pad_width)] + [(0,0)]*(arr.ndim-1),
                   constant_values=pad_value)

def get_bucket(n):
    for b in BUCKETS:
        if n <= b: return b
    return BUCKETS[-1]
```

With 9 bucket sizes covering ~500 to ~130,000 atoms, you get at most 9 JIT compilations (cached after first call). Each compilation takes 0.5–5 seconds; subsequent calls with the same bucket are instant. For a multi-chain complex with 3,500 total atoms, it slots into the 4,096 bucket alongside single-chain proteins of the same size — the JIT cache is shared. Complexes above 65,536 atoms should use the chunked fallback (§10) regardless of GPU size, since gradient storage for the full dense matrix exceeds 12 GB.

### Trajectory processing

```python
@jax.jit
def mean_protection_factors(trajectory, *static_args, **kwargs):
    """trajectory: (n_frames, n_atoms, 3)"""
    def per_frame(coords):
        ln_Pf, _, _ = compute_bv_protection_factors(coords, *static_args, **kwargs)
        return ln_Pf
    # Process 8 frames in parallel, iterate sequentially over batches
    all_ln_Pf = jax.lax.map(per_frame, trajectory, batch_size=8)
    return jnp.mean(all_ln_Pf, axis=0)
```

### Building the sequence exclusion mask (multi-chain aware)

The ±2 residue exclusion must respect chain boundaries — residues from different chains are **never** sequence-neighbours, even if their residue indices happen to be adjacent numerically. The mask is computed once from topology and passed as a static array to all JIT calls.

```python
def build_exclusion_mask(probe_resids, probe_chain_ids,
                         env_resids, env_chain_ids,
                         min_sep=2):
    """Pre-compute static float32 mask. Call once per protein topology.
    
    A contact (probe i, env j) is VALID (mask=1) when:
      - different chains, OR
      - same chain AND |resid_i - resid_j| > min_sep
    Padding atoms (chain_id == -1) are always masked out (mask=0).
    """
    # Shape: (N_probe, N_env)
    same_chain = (probe_chain_ids[:, None] == env_chain_ids[None, :])
    seq_sep = jnp.abs(probe_resids[:, None] - env_resids[None, :])
    
    too_close = same_chain & (seq_sep <= min_sep)
    is_padding = (env_chain_ids[None, :] == -1)   # sentinel for padded atoms
    
    return (~too_close & ~is_padding).astype(jnp.float32)
```

When padding atoms to a bucket size, assign them `chain_id = -1` and `resid = -9999` — the mask construction above automatically zeroes them out. For multi-chain complexes, concatenate all chains into flat arrays before calling this function; the resulting `(N_probe_total, N_env_total)` mask correctly handles all cross-chain and intra-chain pairs without any per-chain logic inside the hot loop.

### Safe fallback for very large systems

If a protein exceeds the memory budget for the dense approach (~50K+ atoms), fall back to probe-chunked computation:

```python
@jax.jit
def chunked_contacts(coords, probe_idx, env_idx, excl_mask, r0, b):
    env_coords = coords[env_idx]
    CHUNK = 256
    
    @jax.checkpoint  # recompute forward during backward, saves memory
    def process_chunk(_, chunk_data):
        p_idx, mask_chunk = chunk_data
        p_coords = coords[p_idx]
        dist = pairwise_dist_matmul(p_coords, env_coords)
        counts = jnp.sum(jax.nn.sigmoid(b * (r0 - dist)) * mask_chunk, axis=-1)
        return _, counts
    
    # Reshape probe data into chunks (pre-padded to multiple of CHUNK)
    probe_chunks = probe_idx.reshape(-1, CHUNK)
    mask_chunks = excl_mask.reshape(-1, CHUNK, excl_mask.shape[-1])
    _, all_counts = jax.lax.scan(process_chunk, None, (probe_chunks, mask_chunks))
    return all_counts.reshape(-1)
```

With `jax.checkpoint`, gradient memory stays at O(CHUNK × N_env) regardless of how many chunks are processed — the forward activations are recomputed during the backward pass at ~2× the forward compute cost. **For multi-chain systems**, the chunking operates over the flat concatenated probe list; no chain-awareness is needed inside this function since the exclusion mask already encodes chain boundaries.

## Conclusion

The BV HDX model's asymmetric geometry — a small probe set querying a larger environment — makes the dense pairwise matrix the clear winner over neighbor-list approaches. **For the largest single-chain proteins (~16,000 atoms), the full dense computation with gradients consumes ~400 MB** — well within modern GPU capacity and with zero risk of neighbor-list overflow, no complex data structure management, and trivial JIT compilation. The sigmoid switching function (Wan et al. JCTC 2020) is the recommended default: it is numerically unconditionally stable, directly comparable to published parameter grids, and equivalent to the tanh switch via `sigmoid(b·Δr) = 0.5*(1 + tanh(b·Δr/2))`. The Wan model extends the standard BV equation with a cooperativity offset β₀ — set `beta_0=0.0` to recover plain BV — which absorbs correlation between N_c and N_h and improves fit to experimental data. Treating `(x_c, x_h, b)` as hyperparameters rather than fixed constants is best practice — caching raw per-frame distances enables rapid post-hoc grid search at negligible cost. The Wan model adds a cooperativity intercept β₀ to the standard BV equation to account for correlation between the N_c and N_h terms; setting β₀ = 0 recovers the original BV model exactly, so both are supported by the same implementation. Multi-chain systems require no architectural changes beyond a chain-aware exclusion mask and concatenated flat atom arrays; the power-of-2 bucket strategy covers total system size regardless of chain count. The key implementation details — the double-where pattern for NaN-safe gradients, epsilon in `sqrt` for zero-distance stability, `lax.map` with `batch_size` for trajectory processing, and power-of-2 bucket padding for JIT caching — are the critical engineering patterns that make the difference between a correct and an incorrect differentiable implementation.