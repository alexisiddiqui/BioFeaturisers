# BioFeaturisers — Internal Regression & Module Test Design

## Conventions

- **Regression tests** verify known-answer outputs and mathematical invariants.
- **Module tests** verify correctness of individual components in isolation.
- All tests use small synthetic inputs (≤100 atoms) for speed; no external PDB files.
- Gradient correctness is verified via finite-difference (FD) checks against `jax.grad`.
- Tolerances: `atol=1e-5` for float32 forward values, `rtol=1e-3` for FD gradient checks.

---

## 1. `core/safe_math.py`

### 1.1 `safe_sinc` — forward values

| Input | Expected | Rationale |
|-------|----------|-----------|
| `qr = 0.0` | `1.0` | sinc(0) = 1 by definition |
| `qr = 1e-10` | `≈ 1.0` | Deep in Taylor branch |
| `qr = 1e-8` | `≈ 1.0` | Boundary of Taylor/normal switch |
| `qr = π` | `0.0` (to float32 tol) | sin(π)/π = 0 |
| `qr = 2π` | `0.0` | sin(2π)/(2π) = 0 |
| `qr = 0.1` | `sin(0.1)/0.1 = 0.99833...` | Normal branch, verify against `np.sinc(0.1/π)` |
| `qr = 100.0` | `sin(100)/100` | Large argument, normal branch |
| Array of `[0.0, 1e-9, 0.01, 1.0, π, 10.0, 100.0]` | Elementwise reference | Vectorised path matches scalar |

### 1.2 `safe_sinc` — gradient correctness

| Test | Method | Pass criterion |
|------|--------|----------------|
| `grad(safe_sinc)(0.0)` | Analytic | `== 0.0`, not NaN |
| `grad(safe_sinc)(1e-10)` | FD vs autodiff | `rtol < 1e-3` |
| `grad(safe_sinc)(π)` | FD vs autodiff | `rtol < 1e-3` |
| `grad(safe_sinc)(0.1)` | Analytic: `(cos(0.1) - sinc(0.1))/0.1` | Match to `atol=1e-5` |
| `grad(safe_sinc)(100.0)` | FD vs autodiff | `rtol < 1e-3` |
| Batch gradient: `grad(lambda x: sum(safe_sinc(x)))(array)` | No NaN in output | All elements finite |

### 1.3 `safe_sinc` — custom VJP consistency

```
# Verify custom VJP matches naive double-where autodiff
qr_test = jnp.array([0.0, 1e-9, 0.01, 0.5, 1.0, π, 5.0, 50.0])
g_custom = jax.grad(lambda x: jnp.sum(safe_sinc(x)))(qr_test)
g_naive  = jax.grad(lambda x: jnp.sum(safe_sinc_no_custom_vjp(x)))(qr_test)
assert allclose(g_custom, g_naive, atol=1e-5)
```

### 1.4 `safe_sqrt` / `safe_sqrt_sym`

| Test | Expected |
|------|----------|
| `safe_sqrt(0.0)` | `sqrt(1e-10) ≈ 1e-5` |
| `safe_sqrt(4.0)` | `2.0` |
| `safe_sqrt_sym(zeros(4,4))` | `zeros(4,4)` |
| `safe_sqrt_sym(eye(4))` | `eye(4)` (diagonal 1s, off-diag 0s) |
| `grad(sum(safe_sqrt_sym(X)))(zeros(4,4))` | All finite, no NaN |
| `grad(sum(safe_sqrt_sym(X)))(eye(4) * 4.0)` | Diagonal: `1/(2*2)=0.25`; off-diag: `0.0` |

### 1.5 `safe_mask`

```
mask = jnp.array([True, False, True, False])
x    = jnp.array([4.0, 0.0, 9.0, 0.0])
# fn = sqrt — would produce inf gradient at 0.0 without safe_mask
result = safe_mask(mask, jnp.sqrt, x, placeholder=0.0)
assert result == [2.0, 0.0, 3.0, 0.0]
grad_result = jax.grad(lambda x: jnp.sum(safe_mask(mask, jnp.sqrt, x)))(x)
assert all_finite(grad_result)
```

### 1.6 `diagonal_self_pairs`

```
ff = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
result = diagonal_self_pairs(ff)  # sum of squares along axis=0
assert result == [1+9+25, 4+16+36] == [35.0, 56.0]
```

---

## 2. `core/pairwise.py`

### 2.1 `dist_matrix_asymmetric` — known geometry

**Two-point test:**
```
probe = jnp.array([[0.0, 0.0, 0.0]])
env   = jnp.array([[3.0, 4.0, 0.0]])
dist = dist_matrix_asymmetric(probe, env)
assert dist[0, 0] ≈ 5.0  # 3-4-5 triangle
```

**Collinear points:**
```
probe = jnp.array([[0,0,0], [1,0,0]])
env   = jnp.array([[5,0,0], [10,0,0]])
# Expected: [[5, 10], [4, 9]]
```

**Symmetry:** `dist(A, B)` should equal `dist(B, A).T`.

**Self-distance:** `dist_matrix_asymmetric(X, X)` diagonal should be `≈ sqrt(1e-10) ≈ 1e-5` (epsilon floor), off-diagonal should match `pdist`.

### 2.2 `dist_matrix_asymmetric` — gradient

```
probe = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
env   = jnp.array([[3.0, 4.0, 0.0]], dtype=jnp.float32)
# d(dist)/d(probe_x) at this geometry = (0-3)/5 = -0.6
g = jax.grad(lambda p: dist_matrix_asymmetric(p, env).sum())(probe)
assert g[0, 0] ≈ -0.6
assert g[0, 1] ≈ -0.8
assert g[0, 2] ≈ 0.0
```

### 2.3 `dist_matrix_block` — diagonal zeros

```
coords = random(5, 3)
dsq = dist_matrix_block(coords, coords)
assert diag(dsq) == zeros(5)     # exact zeros on diagonal
assert dsq == dsq.T              # symmetric
assert all(dsq >= 0)
```

### 2.4 `dist_from_sq_block` — zero handling

```
dsq = jnp.zeros((3, 3))
d = dist_from_sq_block(dsq)
assert d == zeros(3,3)
g = jax.grad(lambda x: dist_from_sq_block(x).sum())(dsq)
assert all_finite(g)  # no NaN/Inf
```

### 2.5 `dist_matrix_block` vs `dist_matrix_asymmetric` cross-check

For the same coordinate pair, the broadcast-based block distance (after sqrt) should match the matmul-based asymmetric distance within float32 tolerance:

```
coords_a = random(20, 3)
coords_b = random(30, 3)
d_asym  = dist_matrix_asymmetric(coords_a, coords_b)
d_block = dist_from_sq_block(dist_matrix_block(coords_a, coords_b))
assert allclose(d_asym, d_block, atol=1e-4)
```

Note: tolerance is relaxed because the two methods have different numerical pathways (matmul identity vs broadcast subtraction). Discrepancy > 1e-3 at any element is a bug.

---

## 3. `core/switching.py`

### 3.1 Midpoint identity

All three switches must return `0.5` at `r = r0`:

```
for r0 in [2.4, 6.5, 10.0]:
    assert sigmoid_switch(r0, r0, b=10.0) ≈ 0.5
    assert tanh_switch(r0, r0, k=5.0) ≈ 0.5
    assert rational_switch(r0, r0, n=6, m=12) ≈ 0.5
```

### 3.2 Sigmoid ↔ tanh equivalence

```
r = jnp.linspace(0, 15, 200)
for b in [3, 5, 10, 20]:
    s = sigmoid_switch(r, r0=6.5, b=b)
    t = tanh_switch(r, r0=6.5, k=b/2)
    assert allclose(s, t, atol=1e-6)
```

### 3.3 Monotonicity

All switches must be monotonically non-increasing with distance:

```
r = jnp.linspace(0, 20, 500)
s = sigmoid_switch(r, r0=6.5, b=10.0)
assert all(diff(s) <= 1e-7)  # non-increasing
```

### 3.4 Asymptotic limits

```
assert sigmoid_switch(0.0, 6.5, 10.0) ≈ 1.0   (within 1e-6)
assert sigmoid_switch(20.0, 6.5, 10.0) ≈ 0.0   (within 1e-6)
```

### 3.5 Gradient smoothness

```
r = jnp.linspace(0, 15, 200)
g = jax.vmap(jax.grad(lambda r: sigmoid_switch(r, 6.5, 10.0)))(r)
assert all_finite(g)
assert max(abs(g)) at r ≈ 6.5  # steepest gradient at midpoint
```

### 3.6 `rational_switch` singularity handling

```
# At exactly r = r0, the function should return n/m, not NaN
assert rational_switch(6.5, 6.5, n=6, m=12) ≈ 0.5
g = jax.grad(lambda r: rational_switch(r, 6.5))(jnp.float32(6.5))
assert is_finite(g)
```

### 3.7 `apply_switch_grid` — shape and consistency

```
dist = random(10, 50)           # 10 probes, 50 env atoms
mask = ones(10, 50)
r0s  = jnp.array([5.0, 6.5, 8.0])
bs   = jnp.array([5.0, 10.0])
result = apply_switch_grid(dist, mask, r0s, bs)
assert result.shape == (3, 2, 10)   # (K, L, N_probe)

# Cross-check: grid result at (r0=6.5, b=10) matches direct sigmoid call
direct = jnp.sum(jax.nn.sigmoid(10.0 * (6.5 - dist)) * mask, axis=-1)
assert allclose(result[1, 1, :], direct, atol=1e-5)
```

---

## 4. `core/ensemble.py`

### 4.1 Single structure passthrough

```
fn = lambda c: jnp.sum(c, axis=0)   # (N,3) → (3,)
coords = random(10, 3)
result = apply_forward(fn, coords, weights=None)
assert result.shape == (3,)
assert allclose(result, jnp.sum(coords, axis=0))
```

### 4.2 Uniform trajectory mean

```
traj = random(5, 10, 3)   # 5 frames
result = apply_forward(fn, traj, weights=None, batch_size=2)
expected = jnp.mean(jax.vmap(fn)(traj), axis=0)
assert allclose(result, expected, atol=1e-6)
```

### 4.3 Weighted trajectory mean

```
w = jnp.array([0.5, 0.3, 0.1, 0.05, 0.05])
result = apply_forward(fn, traj, weights=w, batch_size=2)
expected = jnp.sum(w[:, None] * jax.vmap(fn)(traj), axis=0)
assert allclose(result, expected, atol=1e-6)
```

### 4.4 Gradient through trajectory

```
loss = lambda c: jnp.sum(apply_forward(fn, c, weights=None, batch_size=2))
g = jax.grad(loss)(traj)
assert g.shape == traj.shape
assert all_finite(g)
```

---

## 5. HDX module — `hdx/forward.py`

### 5.1 Synthetic protein: known contact geometry

Construct a minimal synthetic system where contact counts are analytically predictable.

**Setup: 3 residues, collinear along x-axis, spaced 3.8 Å apart (ideal Cα spacing).**

```
# Residue 1: amide N at (0,0,0), amide H at (0,1.01,0)
# Residue 2: amide N at (3.8,0,0), amide H at (3.8,1.01,0)
# Residue 3: amide N at (7.6,0,0), amide H at (7.6,1.01,0)
# Heavy atoms: one per residue at the N position (plus extras at ±1 Å)
# Backbone O: at (1.2,0,0), (5.0,0,0), (8.8,0,0)
```

With `x_c = 6.5 Å`, `b = 50` (very sharp → near-hard cutoff):
- Residue 1 N should see heavy atoms from residues 1 and 2 (within 6.5 Å), not residue 3 (7.6 Å away, after seq-sep exclusion)
- Verify `Nc` counts match hand-computed sigmoid sums

With `x_h = 2.4 Å`, `b = 50`:
- Residue 2 H at (3.8, 1.01, 0) is ~2.9 Å from backbone O at (1.2, 0, 0) → sigmoid ≈ 0
- Place a backbone O at (3.8, 2.0, 0) → distance ~1.0 Å from H → sigmoid ≈ 1

### 5.2 Exclusion mask correctness

```
# 5 residues, single chain, min_sep=2
probe_resids   = [1, 2, 3, 4, 5]
probe_chains   = ['A','A','A','A','A']
env_resids     = [1, 1, 2, 3, 4, 5, 5]  # multiple atoms per residue
env_chains     = ['A','A','A','A','A','A','A']

mask = build_exclusion_mask(probe_resids, probe_chains, env_resids, env_chains, min_sep=2)

# Probe residue 3 should EXCLUDE env atoms with resid in {1,2,3,4,5} where |3-resid|≤2
# So exclude resid 1,2,3,4,5: wait — |3-1|=2 → excluded, |3-2|=1 → excluded,
# |3-3|=0 → excluded, |3-4|=1 → excluded, |3-5|=2 → excluded
# Residue 3 excludes env resids {1,2,3,4,5} — all of them (seq_sep ≤ 2)
# This is correct for a 5-residue peptide — very few valid contacts
```

### 5.3 Multi-chain exclusion mask

```
probe_resids = [1, 2, 1, 2]
probe_chains = ['A','A','B','B']
env_resids   = [1, 2, 3, 1, 2, 3]
env_chains   = ['A','A','A','B','B','B']

mask = build_exclusion_mask(..., min_sep=2)

# Probe A:1 → env A:1 excluded (sep=0), env A:2 excluded (sep=1), env A:3 allowed (sep=2... wait, |1-3|=2, so excluded if min_sep=2)
# But ALL cross-chain contacts allowed:
# Probe A:1 → env B:1, B:2, B:3 all allowed (different chain)
# Verify these specific entries
```

### 5.4 Padding atom exclusion

```
# Padding atoms have chain_id = -1
env_chains = ['A', 'A', -1, -1]
mask = build_exclusion_mask(...)
assert mask[:, 2:] == 0.0   # all padding env atoms masked out
```

### 5.5 Zero coordinates → zero contacts

With all coordinates at origin and proper self-exclusion mask, contacts should be 0 (all distances are 0, which is within cutoff — but exclusion mask removes self-residue). Verify this edge case doesn't produce NaN.

### 5.6 Rigid translation invariance

```
coords1 = random_protein_coords()
coords2 = coords1 + jnp.array([100.0, -50.0, 200.0])  # translate
result1 = hdx_forward(coords1, features, config)
result2 = hdx_forward(coords2, features, config)
assert allclose(result1['Nc'], result2['Nc'], atol=1e-5)
assert allclose(result1['Nh'], result2['Nh'], atol=1e-5)
```

### 5.7 Rigid rotation invariance

```
R = random_rotation_matrix()
coords_rot = coords @ R.T
result_orig = hdx_forward(coords, features, config)
result_rot  = hdx_forward(coords_rot, features_with_rotated_H, config)
# Note: amide H positions must also be rotated consistently
assert allclose(result_orig['ln_Pf'], result_rot['ln_Pf'], atol=1e-5)
```

### 5.8 Protection factor gradient

```
loss = lambda c: jnp.sum(hdx_forward(c, features, config)['ln_Pf'])
g = jax.grad(loss)(coords)
assert g.shape == coords.shape
assert all_finite(g)

# FD check on a random direction
direction = random_like(coords)
direction /= norm(direction)
eps = 1e-4
fd = (loss(coords + eps*direction) - loss(coords - eps*direction)) / (2*eps)
ad = jnp.sum(g * direction)
assert abs(fd - ad) / (abs(ad) + 1e-8) < 1e-2
```

### 5.9 `beta_0` intercept

```
# With beta_0=0: standard BV
result_bv = hdx_forward(coords, features, config_with_beta0_zero)
# With beta_0=1.5: Wan extension
result_wan = hdx_forward(coords, features, config_with_beta0_1_5)
assert allclose(result_wan['ln_Pf'], result_bv['ln_Pf'] + 1.5, atol=1e-6)
```

### 5.10 Trajectory consistency

```
# Single-frame trajectory should match single-structure result
traj = coords[None, :, :]  # (1, N, 3)
result_single = hdx_forward(coords, features, config)
result_traj   = hdx_forward_trajectory(traj, features, config)
assert allclose(result_single['ln_Pf'], result_traj['ln_Pf'], atol=1e-6)
```

---

## 6. HDX module — `hdx/hdxrate.py`

### 6.1 Per-chain N-terminus zeroing

```
# Two chains: A (5 residues), B (3 residues)
kint = compute_kint(topology_AB, pH=7.0, temperature=298.15)
# A:1 should be NaN (N-term of chain A)
# B:1 should be NaN (N-term of chain B)
# A:2 should NOT be NaN (not N-term)
assert isnan(kint[idx_of("A:1")])
assert isnan(kint[idx_of("B:1")])
assert not isnan(kint[idx_of("A:2")])
```

### 6.2 Proline handling

```
# Sequence: "APGK" → residue 2 is Pro
kint = compute_kint(topology_APGK, ...)
assert isnan(kint[1])   # Pro
assert not isnan(kint[2])  # Gly, exchangeable
```

### 6.3 Concatenation guard

```
# Verify that calling HDXrate on concatenated "AAAABBB" gives DIFFERENT results
# than calling per-chain "AAAA" + "BBB"
# Because "AAAA"[0] → 0.0 and "BBB"[0] → 0.0 (two N-termini)
# But "AAAABBB"[0] → 0.0 and [4] → non-zero (B:1 is NOT treated as N-term)
```

### 6.4 Uptake curve sanity

```
# Fully protected residue (ln_Pf = 10) at t=1s with kint=1.0 s⁻¹
# k_obs = kint * exp(-ln_Pf) = exp(-10) ≈ 4.5e-5
# D(1s) = 1 - exp(-4.5e-5) ≈ 4.5e-5
# Fully exposed residue (ln_Pf = 0) at t=1s with kint=1.0
# D(1s) = 1 - exp(-1) ≈ 0.632
```

---

## 7. SAXS module — `saxs/form_factors.py`

### 7.1 Waasmaier-Kirfel at q=0

At q=0 (s=0), the form factor equals the sum of `a` coefficients plus `c`, which should equal the atomic number Z for neutral atoms:

```
for element in ['C', 'N', 'O', 'S', 'H']:
    ff_q0 = compute_ff_table(a[element], b[element], c[element], q=jnp.array([0.0]))
    assert abs(ff_q0[0, 0] - Z[element]) < 0.1
    # Tolerance 0.1 because WK parameterisation has small discrepancies at q=0
```

### 7.2 Form factor positivity

Vacuum form factors should be positive for all q in typical SAXS range:

```
q = jnp.linspace(0.01, 0.5, 300)
for element in ['C', 'N', 'O', 'S']:
    ff = compute_ff_table(a[element], b[element], c[element], q)
    assert all(ff > 0)
```

### 7.3 Form factor monotonic decay

Vacuum form factors decay monotonically with q for single elements:

```
q = jnp.linspace(0.0, 0.5, 300)
ff = compute_ff_table(a['C'], b['C'], c['C'], q)
assert all(diff(ff[0, :]) <= 0)  # monotonically non-increasing
```

### 7.4 Excluded volume form factor

```
# f_excl(q=0) = ρ₀ * V (number of displaced solvent electrons)
# For carbon: V ≈ 16.44 Å³, ρ₀ = 0.334 → f_excl(0) ≈ 5.49
f_excl = compute_excl_ff(V_carbon, rho0=0.334, q=jnp.array([0.0]))
assert abs(f_excl[0] - 0.334 * 16.44) < 0.1
```

### 7.5 Gather consistency

```
# ff_table[atom_type_indices] should produce correct per-atom form factors
types = jnp.array([0, 1, 0, 2])  # C, N, C, O
table = compute_ff_table(...)     # (3, Q) for C, N, O
gathered = table[types]           # (4, Q)
assert gathered[0] == gathered[2]  # both carbon
assert gathered[0] != gathered[1]  # C ≠ N
```

---

## 8. SAXS module — `saxs/debye.py`

### 8.1 Two-atom Debye sum (analytic reference)

The simplest non-trivial case: two atoms separated by distance `d`.

```
# I(q) = F₁² + F₂² + 2·F₁·F₂·sinc(q·d)
coords = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # d = 5 Å
ff = jnp.array([[1.0], [1.0]])  # constant form factors (Q=1, f=1)
q = jnp.array([0.3])

I_calc = debye_chunked(coords, ff, q, chunk_size=2)
I_expected = 1.0 + 1.0 + 2 * 1.0 * 1.0 * jnp.sinc(0.3 * 5.0 / jnp.pi)
# Note: jnp.sinc uses the normalised convention sinc(x) = sin(πx)/(πx)
# Our sinc is unnormalised: sin(qr)/(qr)
I_expected_unnorm = 1.0 + 1.0 + 2 * jnp.sin(0.3 * 5.0) / (0.3 * 5.0)
assert allclose(I_calc, I_expected_unnorm, atol=1e-5)
```

### 8.2 Single atom: I(q) = F(q)²

```
coords = jnp.array([[0.0, 0.0, 0.0]])
ff = jnp.array([[2.0, 3.0, 4.0]])  # Q=3
q = jnp.array([0.1, 0.2, 0.3])
I_calc = debye_chunked(coords, ff, q, chunk_size=1)
assert allclose(I_calc, jnp.array([4.0, 9.0, 16.0]))
```

### 8.3 N identical atoms at same position: I(q) = (N·F)²

```
N = 10
coords = jnp.zeros((N, 3))
ff = jnp.ones((N, 1)) * 2.0
q = jnp.array([0.1])
I_calc = debye_chunked(coords, ff, q, chunk_size=4)
# All distances = 0, sinc(0) = 1
# I = sum_ij F_i * F_j * 1 = (sum F_i)² = (N*2)² = 400
assert allclose(I_calc, jnp.array([400.0]), atol=1e-3)
```

### 8.4 I(q=0) = (Σ Fᵢ(0))²

At q=0, sinc(0)=1 for all pairs, so:

```
q = jnp.array([0.0])  # or very small, e.g. 1e-6
ff_q0 = jnp.array([[6.0], [7.0], [8.0]])  # 3 atoms
coords = random(3, 3) * 10
I_calc = debye_chunked(coords, ff_q0, q)
I_expected = (6 + 7 + 8)**2  # = 441
assert allclose(I_calc, jnp.array([441.0]), atol=1e-2)
```

### 8.5 Translation invariance

```
coords2 = coords + jnp.array([1000.0, -500.0, 2000.0])
I1 = debye_chunked(coords, ff, q)
I2 = debye_chunked(coords2, ff, q)
assert allclose(I1, I2, atol=1e-4)
```

### 8.6 Rotation invariance

```
R = random_rotation_matrix()
coords_rot = coords @ R.T
I_orig = debye_chunked(coords, ff, q)
I_rot  = debye_chunked(coords_rot, ff, q)
assert allclose(I_orig, I_rot, atol=1e-4)
```

### 8.7 Padding atoms contribute nothing

```
# Add 5 padding atoms with zero form factors
coords_pad = jnp.concatenate([coords, jnp.zeros((5, 3))])
ff_pad = jnp.concatenate([ff, jnp.zeros((5, Q))])
I_orig = debye_chunked(coords, ff, q, chunk_size=4)
I_pad  = debye_chunked(coords_pad, ff_pad, q, chunk_size=4)
assert allclose(I_orig, I_pad, atol=1e-5)
```

### 8.8 Symmetry: block weight consistency

The full-loop with `weight = 2.0` for off-diagonal and `1.0` for diagonal should match an explicit N×N dense computation:

```
# Small enough for dense: N=20, Q=5
coords = random(20, 3) * 10
ff = random(20, 5) + 0.5
q = jnp.linspace(0.01, 0.5, 5)

I_chunked = debye_chunked(coords, ff, q, chunk_size=5)  # 4 chunks
I_dense = dense_debye(coords, ff, q)  # explicit N×N reference
assert allclose(I_chunked, I_dense, atol=1e-3)
```

Where `dense_debye` is a simple reference:
```python
def dense_debye(coords, ff, q):
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
    qr = q[None, None, :] * dist[:, :, None]
    sinc_vals = safe_sinc(qr)
    return jnp.sum(ff[:, None, :] * ff[None, :, :] * sinc_vals, axis=(0, 1))
```

### 8.9 Gradient through Debye sum

```
loss = lambda c: jnp.sum(debye_chunked(c, ff, q, chunk_size=4))
g = jax.grad(loss)(coords)
assert all_finite(g)

# FD check
direction = random_like(coords); direction /= norm(direction)
eps = 1e-4
fd = (loss(coords + eps*direction) - loss(coords - eps*direction)) / (2*eps)
ad = jnp.sum(g * direction)
assert abs(fd - ad) / (abs(ad) + 1e-8) < 5e-2
```

### 8.10 Chunk size independence

```
# Same result regardless of chunk_size
for cs in [2, 4, 8, 16, 20]:
    I = debye_chunked(coords, ff, q, chunk_size=cs)
    assert allclose(I, I_ref, atol=1e-4)
```

---

## 9. SAXS module — `saxs/foxs.py` (six partial sums)

### 9.1 Single form-factor type reduces to simple Debye

When `ff_excl = 0` and `ff_water = 0` (or `c1=0, c2=0`), only `Iaa` contributes and should match the standard Debye sum:

```
partials = saxs_six_partials(coords, features_vac_only)
I_combined = saxs_combine(partials, c1=0.0, c2=0.0)
I_debye = debye_chunked(coords, ff_vac, q)
assert allclose(I_combined, I_debye, atol=1e-4)
```

### 9.2 Partial sum symmetry

Each partial sum `Ixy` should be invariant under swapping atom indices (it's a symmetric bilinear form):

```
# Shuffle atom ordering
perm = random_permutation(N)
coords_perm = coords[perm]
ff_vac_perm = ff_vac[perm]
# ... permute all form factor arrays
partials_orig = saxs_six_partials(coords, features)
partials_perm = saxs_six_partials(coords_perm, features_perm)
assert allclose(partials_orig, partials_perm, atol=1e-4)
```

### 9.3 `combine_partials` polynomial identity

```
# Verify: I = Iaa - c1*Iac + c1²*Icc + c2*Ias - c1*c2*Ics + c2²*Iss
partials = jnp.array([[10.0], [3.0], [1.0], [4.0], [2.0], [0.5]])  # (6, 1)
c1, c2 = 1.05, 2.0
I = saxs_combine(partials, c1, c2)
expected = 10 - 1.05*4 + 1.05**2*3 + 2.0*2 - 1.05*2.0*0.5 + 2.0**2*1.0
assert allclose(I, jnp.array([expected]), atol=1e-5)
```

### 9.4 c1/c2 gradient from partials

```
# Analytic: dI/dc1 = -Iac + 2*c1*Icc - c2*Ics
dI_dc1 = jax.grad(lambda c1: saxs_combine(partials, c1, c2).sum())(c1)
expected_dc1 = -4.0 + 2*1.05*3.0 - 2.0*0.5
assert allclose(dI_dc1, expected_dc1, atol=1e-5)
```

### 9.5 Six partials gradient through coordinates

```
loss = lambda c: saxs_six_partials(c, features).sum()
g = jax.grad(loss)(coords)
assert all_finite(g)
# FD spot-check
```

### 9.6 Diagonal contribution extraction

The diagonal (i=j) contribution to each partial sum should equal:
- `Iaa_diag = sum(ff_vac²)`
- `Icc_diag = sum(ff_excl²)`
- etc.

Verify that when all atoms are at the same position (all distances = 0, sinc = 1), the result equals `(Σ ff_eff)²`, which decomposes correctly into the six partials.

---

## 10. SAXS hydration — `saxs/hydration.py`

### 10.1 Grid search finds minimum

```
# Create synthetic I_exp from known c1, c2
partials = saxs_six_partials(coords, features)
c1_true, c2_true = 1.05, 2.0
I_exp = saxs_combine(partials, c1_true, c2_true)
sigma = jnp.ones_like(I_exp) * 0.01

# Grid search should recover c1_true, c2_true
c1_fit, c2_fit = grid_search_c1c2(partials, I_exp, sigma,
                                    c1_range=(0.95, 1.12), c2_range=(0.0, 4.0))
assert abs(c1_fit - c1_true) < 0.02  # within grid step
assert abs(c2_fit - c2_true) < 0.25  # within grid step
```

### 10.2 Scale factor correctness

For each (c1, c2) candidate, the optimal scale factor `a` minimising χ² is analytic:
`a = Σ(I_calc·I_exp/σ²) / Σ(I_calc²/σ²)`. Verify this is applied correctly.

---

## 11. Integration tests

### 11.1 `featurise → forward` round-trip (HDX)

```
# Build features from a synthetic AtomArray (or minimal PDB-like structure)
features = hdx.featurise(atom_array, config=HDXConfig())
result = hdx.forward(coords, features, config=HDXConfig())
assert 'Nc' in result and 'Nh' in result and 'ln_Pf' in result
assert result['Nc'].shape == (n_exchangeable_residues,)
assert all_finite(result['ln_Pf'])
```

### 11.2 `featurise → forward` round-trip (SAXS)

```
features = saxs.featurise(atom_array, config=SAXSConfig(n_q=50))
I_q = saxs.forward(coords, features, config=SAXSConfig(n_q=50))
assert I_q.shape == (50,)
assert all(I_q > 0)   # intensity is non-negative
assert all_finite(I_q)
```

### 11.3 `save → load` fidelity

```
features.save("/tmp/test_protein")
features_loaded = HDXFeatures.load("/tmp/test_protein")
assert allclose(features.excl_mask_c, features_loaded.excl_mask_c)
assert features.res_keys == features_loaded.res_keys
# Forward pass on loaded features should match original
result_orig = hdx.forward(coords, features, config)
result_load = hdx.forward(coords, features_loaded, config)
assert allclose(result_orig['ln_Pf'], result_load['ln_Pf'])
```

### 11.4 `predict` end-to-end

```
result = hdx.predict(atom_array, config=HDXConfig())
# Should return same result as featurise + forward
features = hdx.featurise(atom_array, config=HDXConfig())
result2 = hdx.forward(extract_coords(atom_array), features, config=HDXConfig())
assert allclose(result['ln_Pf'], result2['ln_Pf'])
```

### 11.5 Multi-chain consistency

```
# Dimer: chains A and B, identical sequence
# Cross-chain contacts should be included
# Each chain's N-terminus should be excluded
features = hdx.featurise(dimer_atom_array, config=HDXConfig())
assert len(features.kint) == 2 * n_residues_per_chain
assert isnan(features.kint[0])                    # A:1 N-term
assert isnan(features.kint[n_residues_per_chain]) # B:1 N-term
```

---

## 12. Numerical stability stress tests

### 12.1 Very large coordinates

```
coords = random(50, 3) * 1000.0  # atoms 1000 Å apart
I = debye_chunked(coords, ff, q)
assert all_finite(I)
# sinc(q*1000) oscillates rapidly — verify no NaN
```

### 12.2 Very small distances

```
# Two atoms 0.001 Å apart (unphysical but tests edge case)
coords = jnp.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
I = debye_chunked(coords, ff, q)
assert all_finite(I)
g = jax.grad(lambda c: debye_chunked(c, ff, q).sum())(coords)
assert all_finite(g)
```

### 12.3 Large N accumulation (float32)

```
# N=5000 atoms, verify hierarchical summation doesn't drift
# Compare chunk_size=128 (many blocks) vs chunk_size=5000 (single block)
I_128  = debye_chunked(coords_5k, ff_5k, q, chunk_size=128)
I_5000 = debye_chunked(coords_5k, ff_5k, q, chunk_size=5000)
assert allclose(I_128, I_5000, rtol=1e-3)  # relaxed tol for accumulation difference
```

### 12.4 Gradient with `jax.checkpoint`

```
# Verify checkpointed forward produces same gradient as non-checkpointed
loss_no_cp = lambda c: debye_chunked(c, ff, q).sum()
loss_cp = lambda c: jax.checkpoint(debye_chunked)(c, ff, q).sum()
g1 = jax.grad(loss_no_cp)(coords)
g2 = jax.grad(loss_cp)(coords)
assert allclose(g1, g2, atol=1e-5)
```

---

## 13. JIT and shape tests

### 13.1 Recompilation count

```
# Same bucket size should not trigger recompilation
with jax.log_compiles():
    hdx_forward(coords_100, features_100, config)   # compiles (bucket 128)
    hdx_forward(coords_120, features_120, config)   # same bucket — no recompile
    # Verify only 1 compilation logged
```

### 13.2 Power-of-2 padding

```
for n in [100, 500, 1000, 2000, 5000]:
    bucket = get_bucket(n)
    assert bucket >= n
    assert bucket & (bucket - 1) == 0  # is power of 2
```

### 13.3 Static shape consistency

```
# Padded and unpadded should give identical results
coords_pad = pad_to_bucket(coords, 512)
ff_pad = pad_to_bucket(ff, 512)
I_orig = debye_chunked(coords, ff, q, chunk_size=64)
I_pad  = debye_chunked(coords_pad, ff_pad, q, chunk_size=64)
assert allclose(I_orig, I_pad, atol=1e-5)
```

---

## 14. Test fixtures

### Synthetic protein builder

A reusable fixture that constructs a minimal protein-like system with known geometry:

```python
def make_synthetic_protein(n_residues=10, seed=42):
    """Returns coords, topology arrays, and expected contact counts
    for a simple α-helix-like geometry (3.8 Å Cα spacing, 1.5 Å rise)."""
    # Returns: coords (N_atoms, 3), amide_N_idx, amide_H_idx,
    #          heavy_atom_idx, backbone_O_idx, excl_mask_c, excl_mask_h,
    #          expected_Nc_hard_cutoff, expected_Nh_hard_cutoff
```

### Dense Debye reference

```python
def dense_debye_reference(coords, ff, q):
    """Naive O(N²) Debye sum — used only for testing against chunked version.
    Not JIT-compiled, not memory-efficient. Reference implementation."""
```

### Numerical gradient checker

```python
def check_grad(fn, x, eps=1e-4, rtol=1e-2):
    """Compare jax.grad against central finite differences along 5 random directions."""
```