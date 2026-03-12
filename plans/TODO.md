# BioFeaturisers — Phased Implementation Plan

This is a phased implementation plan synthesized from the architectural documents, mathematical formulations, execution policies, and testing guidelines in the `plans/` directory.

## Phase 1: Core Configuration, Environment, and Data Structures
**Goal**: Establish the base configuration classes, environment setup, and key data structures that do not rely on complex JAX compilation or physics.

1. **`env.py`**
   - Implement `ComputeConfig` for JAX device/memory policy selection (GPU vs CPU, `XLA_PYTHON_CLIENT_PREALLOCATE`).
2. **`config.py`**
   - Define dataclasses `HDXConfig` and `SAXSConfig`.
3. **Core Topology Data Structures** (`core/topology.py`, `core/output_index.py`)
   - Create `MinimalTopology` dataclass and its `.to_json()`/`.from_json()` methods.
   - Implement temporary Biotite dummy integration for tests.
   - Create `OutputIndex` dataclass for atom/probe/residue masking and selection routing.
4. **Feature Dataclasses**
   - Define `HDXFeatures` and `SAXSFeatures`.
5. **Basic Test Suite Setup**
   - Initialise testing harness and implement property checks on config classes.

## Phase 2: Math Primitives & Shared Pairwise Engine
**Goal**: Develop the `core/` JAX kernels handling distances, safe gradients, and ensemble batching, which underpin all specific model logic. Ensure type safety using `jaxtyping` annotations and `chex` assertions throughout.

1. **`core/safe_math.py`**
   - Implement `safe_sqrt`, `safe_sqrt_sym`, `safe_mask`, and `diagonal_self_pairs`.
   - Implement `safe_sinc` with the critical custom VJP.
   - Use `jaxtyping` `Float[Array, "..."]` for array signatures.
   - *Test Focus*: `test_safe_math.py` (gradients at origin, limits, finite checks)
2. **`core/pairwise.py`**
   - Implement `dist_matrix_asymmetric` (using matmul identity + $\varepsilon$).
   - Implement `dist_matrix_block` and `dist_from_sq_block` (symmetric blocks).
   - Implement `chunked_dist_apply` for >50K atom fallback.
   - *Test Focus*: `test_pairwise.py` (known geometries, memory shapes).
3. **`core/switching.py`**
   - Implement `sigmoid_switch`, `tanh_switch`, and `rational_switch`.
   - Implement `bv_contact_counts` logic.
   - Implement `apply_switch_grid`.
   - *Test Focus*: Midpoint limits, continuity, sigmoid vs tanh equivalence.
4. **`core/ensemble.py`**
   - Implement `apply_forward` (dispatching to single/trajectory/weighted via `lax.map(batch_size)` and `jax.checkpoint`).

## Phase 3: HDX-MS Module Development
**Goal**: Provide full differentiable tracking of Best-Vendruscolo contacts and protection factors, wrapping the JAX kernels with Biotite parsing and HDXrate.

1. **`hdx/forward.py`**
   - Implement `hdx_forward` to produce `Nc`, `Nh`, and `ln_Pf`.
   - Setup `jax.jit` strategy (power-of-2 atom buckets to avoid recompiles).
2. **`hdx/featurise.py`**
   - Use Biotite `AtomArray` to parse topological data to `HDXFeatures`.
   - Implement `build_exclusion_mask` for exact cross/intra-chain logic.
   - Implement analytical amide-H generation from N/CA/C coords (if H atoms are missing).
3. **`hdx/hdxrate.py`**
   - Implement `compute_kint` with HDXrate ensuring per-chain evaluation limits.
   - Implement prediction kernel `predict_uptake` over peptides via cached `kint`.
4. **`hdx/predict.py`**
   - Implement `predict` integration wrapper routing `featurise` $\to$ `forward`.
5. **Testing & Validation**
   - Verify `test_hdx_forward.py` comparing the `ln_Pf` against hard BV limits for PDB: `1UBQ` (expect Pearson *r* > 0.95).

## Phase 4: SAXS Module Development
**Goal**: Build out the FoXS model utilizing six-partial sums across chunked interactions without materialising the $N \times N \times Q$ dimension.

1. **`saxs/form_factors.py`**
   - Implment WK 5-Gaussian table gathers ($f^{\text{vac}}$).
   - Implement Fraser Gaussian sphere generation ($f^{\text{excl}}$).
   - Implement $f^{\text{water}} = 2 f_H + f_O - f^{\text{excl}}_{\text{water}}$ formulation.
2. **`saxs/debye.py`** & **`saxs/foxs.py`**
   - Implement `saxs_six_partials` using `lax.scan` to traverse chunked coordinates and prevent OOM.
   - Implement the O(1) recombination `saxs_combine(partials, c1, c2)`.
   - Integrate `diagonal_self_pairs` cleanly without `sinc(0)` singularity points.
3. **`saxs/hydration.py`**
   - Implement $c_{1}$ / $c_{2}$ analytic grid search fitting against $I^{\text{exp}}$.
4. **`saxs/featurise.py`** & **`saxs/predict.py`**
   - Port Shrake-Rupley SASA calculations via Biotite to produce constant static $S_i$ values.
   - Assemble `saxs.predict` convenience method.
5. **Testing & Validation**
   - Compare `I(q)` against CRYSOL/FoXS outputs on test PDB for $\chi^2 < 1.1$.

## Phase 5: Input / Output and CLI
**Goal**: Support HPC workflow paradigms with standalone processing capabilities.

1. **`io/save.py`** & **`io/load.py`**
   - Implement serialization of features + topologies to `.npz` arrays and matching JSON maps.
2. **`io/formats.py`**
   - Add parsing for generic 3-col SAXS data `.dat` / `.fit` and HDX `.csv` datasets.
   - Handle custom output array index json representations (`_hdx_index.json`, `_saxs_index.json`).
3. **`cli.py`**
   - Build out the Typer application.
   - Register endpoints: `biofeaturisers {hdx/saxs} {featurise/forward/predict}`.
   - Bind `--env` to `ComputeConfig` configuration.

## Phase 6: Sharding, Optimisation, and Benchmarking
**Goal**: Final performance guarantees on scaled multi-GPU hardware.

1. **Memory & Performance Optimisation**
   - Pad outputs rigorously with power-of-2 logic to `chunk_size`. Note padding semantics for XLA caching (use `donate_argnums=(0,)`).
2. **Hardware Sharding Implementation**
   - Refine `lax.map` calls to standard `pmap` or distributed `pjit`. Verify single-host multi-GPU limits.
3. **Internal Benchmarking**
   - Execute tests outlined in `internal_benchmarking.md`: single structures, trajectories, grid searches. Report on latency, peak VRAM limits, scaling.
