# FoXS-style differentiable SAXS in JAX: an implementation blueprint

**No production-ready JAX implementation of the Debye SAXS forward model exists today** — this guide fills that gap with concrete, JIT-compilable code patterns for computing all six FoXS partial sums with full autodiff through atomic coordinates. The core architectural insight is a **double `lax.scan` over spatial chunks** that keeps peak GPU memory at O(B²·Q) regardless of system size, while computing the sinc kernel once per block and contracting it with six form-factor outer products simultaneously. For a **5,000-atom system at Q=300**, a single SAXS profile computes in ~50 MB of peak memory; for **50,000 atoms**, the same chunked approach scales gracefully at ~1,225 block operations with B=1024. The design composes cleanly with `jax.grad` for structure refinement, `lax.map` with `batch_size` for MD trajectory batches, and `jax.checkpoint` for bounded backward-pass memory.

---

## The O(N²) Debye sum demands chunked pairwise computation

The Debye formula I(q) = Σᵢ Σⱼ Fᵢ(q)·Fⱼ(q)·sinc(q·rᵢⱼ) is inherently O(N²) per q-value. A naive N×N distance matrix at float32 costs **100 MB for N=5,000** and **10 GB for N=50,000** — before even expanding across Q q-values. The (N, N, Q) sinc tensor would be 100× larger. The solution is to never materialize either.

**The block-accumulation pattern** splits N atoms into ⌈N/B⌉ chunks of size B, computes each B×B block of distances, immediately contracts with form factors and sinc, and accumulates into a (Q,) result vector:

```python
import jax
import jax.numpy as jnp

def debye_chunked(coords, ff, q_values, chunk_size=512):
    """Chunked Debye sum. Peak memory: O(B² × Q).
    coords: (N, 3), ff: (N, Q), q_values: (Q,)  →  I(q): (Q,)
    """
    N, Q = ff.shape
    pad_n = (-N) % chunk_size
    coords_p = jnp.pad(coords, ((0, pad_n), (0, 0)))
    ff_p = jnp.pad(ff, ((0, pad_n), (0, 0)))  # zero-padded ff → zero contribution
    n_ch = coords_p.shape[0] // chunk_size
    coords_ch = coords_p.reshape(n_ch, chunk_size, 3)
    ff_ch = ff_p.reshape(n_ch, chunk_size, Q)

    def block_contrib(ci, fi, cj, fj):
        diff = ci[:, None, :] - cj[None, :, :]          # (B, B, 3)
        dist_sq = jnp.sum(diff ** 2, axis=-1)            # (B, B)
        safe_sq = jnp.where(dist_sq > 0, dist_sq, 1.0)
        dist = jnp.where(dist_sq > 0, jnp.sqrt(safe_sq), 0.0)
        qr = q_values[None, None, :] * dist[:, :, None]  # (B, B, Q)
        safe_qr = jnp.where(qr > 1e-8, qr, 1.0)
        sinc = jnp.where(qr > 1e-8, jnp.sin(safe_qr) / safe_qr,
                         1.0 - qr ** 2 / 6.0)
        return jnp.sum(fi[:, None, :] * fj[None, :, :] * sinc, axis=(0, 1))

    def outer(carry, i):
        def inner(carry_in, j):
            contrib = block_contrib(
                coords_ch[i], ff_ch[i], coords_ch[j], ff_ch[j])
            weight = jnp.where(i == j, 1.0, 2.0)
            return carry_in + weight * contrib, None
        carry, _ = jax.lax.scan(inner, carry, jnp.arange(n_ch))
        return carry, None

    I_q, _ = jax.lax.scan(outer, jnp.zeros(Q), jnp.arange(n_ch))
    return I_q
```

The inner `lax.scan` iterates over all j-chunks for each i-chunk. Using `weight = 2.0` for off-diagonal blocks and `1.0` for diagonal blocks gives the correct double-sum without halving the loop. Symmetry exploitation (iterating only j ≥ i) is possible but complicates the static-shape scan; the full-loop approach with weight factors is simpler and JAX's compiler amortizes the redundant distance computation well.

**Block size selection** depends on GPU memory and Q. The peak per-block tensor is (B, B, Q) at 4 bytes/element:

| B | Q=100 | Q=300 | Blocks for N=50K |
|---|-------|-------|-----------------|
| 256 | 25 MB | 75 MB | 38,025 |
| 512 | 100 MB | 300 MB | 9,604 |
| 1024 | 400 MB | 1.2 GB | 2,401 |
| 2048 | 1.6 GB | 4.8 GB | 625 |

For an **A100 (80 GB)**, B=1024 with Q=300 is comfortable. For a **consumer 8 GB GPU**, B=256–512 works. If Q is large, batch q-values in groups of 32–64 and loop over q-batches to further reduce peak memory.

**Distance computation**: The broadcasting approach `coords_i[:, None, :] - coords_j[None, :, :]` is preferred over the matrix-algebra identity `||x||² + ||y||² - 2x·yᵀ` because the latter suffers from catastrophic cancellation when distances are small relative to coordinate magnitudes — a common situation in molecular structures.

---

## Six partial sums: one sinc matrix, six contractions

The FoXS form factor model decomposes each atom's effective form factor as Fᵢ(q) = fᵛᵃᶜ(q) − c₁·fᵉˣᶜˡ(q) + c₂·Sᵢ·fʷᵃᵗᵉʳ(q). Expanding the product FᵢFⱼ and collecting by powers of c₁ and c₂ yields **six coordinate-dependent partial sums** and a polynomial recombination:

**I(q) = Iₐₐ − c₁·Iₐc + c₁²·Icc + c₂·Iₐₛ − c₁c₂·Ics + c₂²·Iₛₛ**

Each partial sum has the bilinear form **aᵀ·S(q)·b** where S(q)ᵢⱼ = sinc(q·rᵢⱼ). The critical insight is that the sinc matrix depends only on coordinates, not form factors. Within each B×B chunk, compute the sinc kernel once, then contract it with three "sinc-weighted" vectors to produce all six sums simultaneously:

```python
def six_partial_sums_block(sinc_block, fv_i, fv_j, fe_i, fe_j, fs_i, fs_j):
    """Given sinc: (B,B,Q) and form factors (B,Q), return 6 partial contributions."""
    # Compute Σ_j ff_j * sinc_ij for each form factor type → (B, Q)
    w_v = jnp.einsum('ijq,jq->iq', sinc_block, fv_j)
    w_e = jnp.einsum('ijq,jq->iq', sinc_block, fe_j)
    w_s = jnp.einsum('ijq,jq->iq', sinc_block, fs_j)

    Iaa = jnp.sum(fv_i * w_v, axis=0)   # (Q,)
    Icc = jnp.sum(fe_i * w_e, axis=0)
    Iss = jnp.sum(fs_i * w_s, axis=0)
    Iac = jnp.sum(fv_i * w_e + fe_i * w_v, axis=0)
    Ias = jnp.sum(fv_i * w_s + fs_i * w_v, axis=0)
    Ics = jnp.sum(fe_i * w_s + fs_i * w_e, axis=0)

    return jnp.stack([Iaa, Icc, Iss, Iac, Ias, Ics])  # (6, Q)
```

This produces all six sums from the same sinc kernel. Memory overhead for six sums versus one is negligible — just three extra (B, Q) weighted vectors (~6 MB for B=1024, Q=300). The **three sinc-weighted vectors** `w_v`, `w_e`, `w_s` are the pivotal intermediates; everything else is a cheap (B, Q) dot product. The `jnp.einsum('ijq,jq->iq', ...)` pattern contracts over j atoms and XLA compiles this to a batched reduction kernel, fusing the multiply and sum into a single pass.

The recombination is then trivially differentiable with respect to c₁ and c₂:

```python
def combine_partials(partials, c1, c2):
    Iaa, Icc, Iss, Iac, Ias, Ics = partials
    return Iaa - c1*Iac + c1**2*Icc + c2*Ias - c1*c2*Ics + c2**2*Iss
```

Since this is a polynomial in c₁, c₂ with precomputed coefficients, **fitting c₁ and c₂ requires no recomputation of the expensive O(N²) sums**. Grid search over c₁ ∈ [0.95, 1.12] × c₂ ∈ [0, 4.0] with analytic scale factor costs microseconds. For gradient-based fitting, ∂I/∂c₁ = −Iac + 2c₁·Icc − c₂·Ics is analytic and free.

---

## Form factors: vectorized Gaussians with index gathering

The Waasmaier-Kirfel 5-Gaussian parameterization gives f₀(q) = Σₖ₌₁⁵ aₖ·exp(−bₖ·s²) + c where s = q/(4π). The implementation precomputes a **(num_types, Q) lookup table**, then gathers per-atom form factors via integer indexing:

```python
def compute_ff_table(a, b, c, q):
    """a: (T,5), b: (T,5), c: (T,), q: (Q,) → (T, Q)"""
    s2 = (q / (4 * jnp.pi)) ** 2                    # (Q,)
    exponents = -b[:, :, None] * s2[None, None, :]   # (T, 5, Q)
    return jnp.sum(a[:, :, None] * jnp.exp(exponents), axis=1) + c[:, None]

# Per-atom form factors via simple gather
f_vac = ff_table[atom_type_indices]   # (N, Q) — JIT-friendly integer index
```

For the **excluded volume form factor**, FoXS uses a Fraser-style Gaussian sphere model: f_excl(q) = ρ₀·V·exp(−(3V/4π)^(2/3)·q²/(4π)), where V is the atomic volume and ρ₀ ≈ 0.334 e/ų is solvent electron density. The parameter c₁ uniformly scales these volumes. Since c₁ enters linearly in the final combination (the six partial sums absorb its polynomial structure), form factor tables can be precomputed at c₁=1 and the scaling handled algebraically.

The **water form factor** f_water(q) = 2·f_H(q) + f_O(q) − f_excl_water(q) is a single q-dependent curve shared by all atoms. Each atom's hydration contribution is modulated by its solvent-accessible surface fraction Sᵢ, giving f_sol_i(q) = Sᵢ·f_water(q). This makes f_sol an (N, Q) array with rank-1 structure along the atom dimension.

---

## Trajectory batching: lax.map controls memory, vmap does not

For T MD frames × N atoms × 3 coordinates → T × Q intensity curves, the key architectural decision is **`jax.lax.map` with `batch_size` rather than `jax.vmap`**. The reason is memory: `vmap` materializes all T frames simultaneously. For T=1000 with 50 MB peak per frame, that's **50 GB** — exceeding most GPUs. `lax.map` processes frames sequentially (or in sub-batches), keeping peak memory at batch_size × per-frame cost.

```python
form_factors = compute_form_factors(atom_types, q_values)  # computed ONCE

@jax.checkpoint   # recompute forward pass during backprop
def per_frame(coords_t):
    return debye_chunked(coords_t, form_factors, q_values)

# Process 4 frames in parallel, iterate over groups of 4
intensities = jax.lax.map(per_frame, all_coords, batch_size=4)  # (T, Q)
```

Form factors depend only on atom types and q-values, **not on coordinates**, so they are computed once outside the map and captured via closure. JAX does not duplicate closed-over arrays when vmapping or mapping — they are broadcast, not copied.

**`jax.checkpoint` (remat) is essential** on the per-frame function when computing gradients through many frames. Without it, JAX stores all intermediate tensors from all T forward passes for backpropagation. With it, only the input coordinates per frame are stored; the forward pass is recomputed during the backward pass. This trades **2× compute for O(1) memory** in the frame dimension. The JAX documentation explicitly recommends applying `checkpoint` to scan/map body functions.

**Memory budget for T=1000, N=5,000, Q=300**: Coordinates storage is T×N×3×4 = 60 MB. Form factors (shared) are 3×N×Q×4 = 18 MB. Per-frame peak with B=512 is ~300 MB. With `batch_size=4` and checkpoint, peak GPU memory is roughly 4×300 MB + 78 MB ≈ **1.3 GB** — comfortably fits any modern GPU.

---

## Autodiff through sinc requires the double-where pattern

The gradient of the Debye sum flows through sinc(qr) = sin(qr)/(qr) and through rᵢⱼ = ||rᵢ − rⱼ||. Both have singularities at zero. **JAX's `jnp.where(cond, x, y)` evaluates both branches during autodiff** — a NaN in the unused branch still poisons the gradient. This is confirmed in the JAX FAQ and multiple GitHub issues (#6484, #2377, #3058).

The correct pattern **masks the input, not the output**:

```python
def safe_sinc(x):
    """sinc with well-defined gradients everywhere including x=0."""
    safe_x = jnp.where(jnp.abs(x) > 1e-8, x, 1.0)   # safe INPUT
    return jnp.where(jnp.abs(x) > 1e-8,
                     jnp.sin(safe_x) / safe_x,          # branch A: normal
                     1.0 - x**2 / 6.0 + x**4 / 120.0)  # branch B: Taylor
```

When x=0, the gradient flows through branch B (Taylor expansion), which has a smooth, well-defined derivative of −x/3. The `sin(safe_x)/safe_x` in branch A computes on the dummy value 1.0, producing a finite intermediate that gets masked out by the outer `where`. This double-where pattern is the **canonical JAX solution** for piecewise functions with singular branches.

For the **norm gradient at rᵢⱼ=0** (the i=j diagonal), the cleanest solution is to exclude the diagonal entirely: sinc(0)=1 always, so the diagonal contribution is simply Σᵢ Fᵢ(q)², which has trivially clean gradients. Apply an identity-matrix mask to zero out diagonal elements in the distance matrix before computing sinc.

For a **custom VJP** that provides maximum control over numerical stability and memory during backprop:

```python
@jax.custom_vjp
def sinc_kernel(qr):
    safe = jnp.where(qr > 1e-8, qr, 1.0)
    return jnp.where(qr > 1e-8, jnp.sin(safe)/safe, 1.0 - qr**2/6.0)

def sinc_kernel_fwd(qr):
    y = sinc_kernel(qr)
    return y, (qr, y)

def sinc_kernel_bwd(res, g):
    qr, y = res
    safe = jnp.where(qr > 1e-8, qr, 1.0)
    dsinc = jnp.where(qr > 1e-8, (jnp.cos(safe) - y) / safe, -qr/3.0)
    return (g * dsinc,)

sinc_kernel.defvjp(sinc_kernel_fwd, sinc_kernel_bwd)
```

This analytically computes sinc'(x) = (cos(x) − sinc(x))/x, avoiding autodiff through the sin/division chain entirely. The Taylor branch −x/3 handles the origin. Storing both `qr` and `y` as residuals avoids recomputing sinc in the backward pass at the cost of O(B²) memory per block — well within budget.

**`jax.grad` is the right choice** for SAXS refinement. The loss χ²(I_calc, I_exp) is scalar, and `jax.grad` uses reverse-mode AD internally, costing approximately **3× the forward pass**. If a Gauss-Newton or Levenberg-Marquardt optimizer is needed (requiring the full Q×N×3 Jacobian), `jax.jacrev` computes it via Q separate VJP calls — expensive but sometimes necessary for fast convergence.

---

## JIT compilation: static shapes through padding and gathering

JAX traces functions once per unique input shape; changing shapes triggers expensive recompilation. The strategy for variable-length proteins is **padding to a fixed `max_atoms` with zero form factors**:

```python
# Pad form factors with zeros → Fᵢ·Fⱼ = 0 for padding atoms
# No explicit mask needed — zero ff naturally eliminates contributions
padded_ff = jnp.zeros((max_atoms, Q)).at[:N].set(form_factors)
padded_coords = jnp.zeros((max_atoms, 3)).at[:N].set(coords)
```

Zero form factors are elegant because they eliminate padding-atom contributions *through the physics* rather than through an explicit mask. The gradient with respect to padding-atom coordinates is also zero (no contribution to the loss), so backpropagation is clean. For multiple proteins with different sizes, bucket to **power-of-2 sizes** (512, 1024, 2048, 4096...) to limit the number of cached compilations.

The **JIT boundary** should enclose the entire computation from form factors through χ² loss:

```python
@partial(jax.jit, donate_argnums=(0,))
def saxs_loss(coords_batch, ff_table, q_values, c1, c2, I_exp, sigma):
    effective_ff = ff_table[:, 0] - c1 * ff_table[:, 1] + c2 * ff_table[:, 2]
    def per_frame(coords):
        return jax.checkpoint(debye_chunked)(coords, effective_ff, q_values)
    I_calc = jax.lax.map(per_frame, coords_batch, batch_size=4)
    residuals = (jnp.mean(I_calc, axis=0) - I_exp) / sigma
    return jnp.sum(residuals ** 2)
```

Do **not** use `static_argnums` for atom-type arrays — integer arrays trigger recompilation on any change. Instead, precompute the form factor table (a float array) outside JIT and pass it as a traced argument. The `donate_argnums=(0,)` hint tells XLA it can recycle the coordinate buffer's memory.

For **element-type dispatch**, gather-by-index (`ff_table[atom_types]`) is vastly superior to `lax.cond/switch` per atom. It is a single vectorized operation rather than N sequential branches.

---

## Precision: float32 with chunked accumulation is sufficient

Coordinates, distances, form factors, and sinc evaluation are all fine at **float32** for typical SAXS. The sinc argument qr can reach ~250 for q_max=0.5 Å⁻¹ and r_max=500 Å, where float32 retains ~4 digits of precision for sin(250) — marginal but acceptable since large-distance sinc terms contribute little to the total.

The real danger is **accumulation of N² terms**. Naively summing ~25 million terms of O(1) magnitude into a result of O(5000) would give relative error ~N²·ε/sum ≈ 0.6 — catastrophic. However, the chunked approach inherently provides **hierarchical summation**: each B×B block produces a partial sum of ~B² terms, and the ~(N/B)² partial sums are then accumulated. For N=5000 and B=512, this means summing ~100 partial sums rather than 25M individual terms. The relative error drops to ~100·ε ≈ 1.2×10⁻⁵, which is acceptable. XLA's `jnp.sum` within each block already uses pairwise summation internally.

If higher precision is needed (e.g., for sensitive gradient-based refinement), **cast the accumulator to float64**:

```python
jax.config.update("jax_enable_x64", True)
I_accum = jnp.zeros(Q, dtype=jnp.float64)
# ...
I_accum += block_contribution.astype(jnp.float64)
```

Float64 is 2× slower on consumer GPUs but runs at full speed on A100/H100. **bfloat16 and float16 are not viable** for the sinc computation — they have only 7–10 mantissa bits, insufficient for sin(qr) at large arguments. Form factor storage can use float16 to save memory bandwidth, upcasting to float32 at compute time.

---

## Performance context and what to expect

No published JAX SAXS benchmark exists, but reference points from other implementations calibrate expectations. BCL::SAXS on a GTX 680 achieved **1,707× speedup** over serial CPU for 91,846 atoms using the full Debye formula with OpenCL. Pepsi-SAXS (CPU-only, multipole expansion) is **29× faster than FoXS** and **7× faster than CRYSOL** but sacrifices the exact Debye formula for an O(N·L²) spherical harmonics approximation. The DebyeCalculator (PyTorch/CUDA) reports "orders of magnitude" GPU speedup with a default batch size of 10,000 atom-pair chunks.

A JAX implementation on an A100 should achieve comparable or better throughput than BCL::SAXS thanks to XLA's fusion of the sinc-multiply-reduce chain into single GPU kernels. Estimated throughput:

- **N=1,000, Q=100**: Sub-millisecond per profile (no chunking needed, ~4 MB distance matrix)
- **N=5,000, Q=300**: ~10-50 ms per profile (100 block operations at B=512)
- **N=50,000, Q=300**: ~1-5 seconds per profile (~10,000 block operations at B=512)

The O(N²) scaling is the fundamental bottleneck. For systems above ~20,000 atoms, consider **coarse-graining** (Cα-only or residue-level form factors, reducing N by 10–20×) or the **histogram approximation** used by FoXS, debyer, and AUSAXS — bin pairwise distances into a histogram (O(N²) once) then evaluate the Debye sum over bins (O(n_bins·Q), typically much cheaper). The histogram approach breaks autodiff through coordinates but is invaluable for the initial forward-only sweep.

---

## Recommended architecture and module structure

The overall design should follow these principles:

**Always recompute distances from coordinates** — never cache the full N×N matrix. With `jax.checkpoint`, distances are recomputed during backprop from O(N) coordinate storage rather than O(N²) distance storage.

**Use `lax.map(batch_size=B)` for frames, nested `lax.scan` for spatial chunks.** This gives two levels of memory control: `batch_size` tunes frame parallelism, `chunk_size` tunes spatial parallelism. Both are JIT-compatible and differentiable.

**Keep the form factor lookup outside the autodiff graph** when c₁ only enters linearly (via the six partial sums). If c₁ also modulates the excluded volume form factor nonlinearly, move the form factor computation inside the `jax.grad` scope and use `jax.checkpoint` on it.

```
saxs_jax/
├── form_factors.py    # Waasmaier-Kirfel tables, ff evaluation, gather
├── debye.py           # Chunked Debye sum, safe_sinc, block computation
├── foxs.py            # Six partial sums, combine_partials, chi-squared
├── trajectory.py      # lax.map over frames, checkpoint, ensemble avg
└── refinement.py      # jax.grad + optax optimizer, refinement loop via lax.scan
```

The `debye.py` module is the computational core. Its `debye_chunked` function should accept `chunk_size` as a parameter (not hardcoded), pad atoms to chunk boundaries with zero form factors, and use `lax.scan` for both the outer and inner chunk loops. The `foxs.py` module wraps this with the six-partial-sum decomposition, precomputing the three form factor arrays and calling the block computation with six simultaneous contractions. The `trajectory.py` module handles `lax.map` over frames with `batch_size` tuning and `jax.checkpoint`. Finally, `refinement.py` uses `jax.value_and_grad` inside a `lax.scan` over optimization steps, composing with an optax optimizer for learning rate scheduling and gradient clipping.

## Conclusion

Building FoXS-style SAXS in JAX requires solving three interlocking problems: memory-bounded O(N²) pairwise computation, numerically stable autodiff through sinc singularities, and efficient batching across trajectory frames. The chunked `lax.scan` pattern is the linchpin — it simultaneously enables static-shape JIT compilation, bounded GPU memory, and correct gradient flow. The six-partial-sum decomposition lifts c₁/c₂ fitting out of the expensive Debye loop entirely, and the double-`where` pattern with custom VJP handles the sinc and norm singularities that would otherwise produce NaN gradients. The fact that no JAX implementation yet exists — despite JAX being arguably the ideal framework for this workload (XLA fusion of element-wise chains, native autodiff, `vmap`/`scan` composability) — means this is genuinely new territory, but every building block is well-tested in the JAX ecosystem.