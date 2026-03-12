# Devices, compilation, and distributed execution

## Goals
- Mixed CPU/GPU clusters are first-class. Featurise/topology steps run on CPU/login nodes; forward passes and predict loops run on GPU nodes.
- Minimise host↔device traffic: save/load feature npz + topology JSON, keep coordinates on-device across batches.
- Prefer reproducible, deterministic runs where feasible; allow opt-in speed via lower precision.

## Device/precision policy
- Default: GPU when available (`jax_default_device`). CPU is fallback for topology/featurise and small tests.
- Precision: keep `jax_enable_x64=False` for topology prep; allow fp32 in forward kernels unless gradients demand fp64. Avoid mixed precision until kernels are validated.
- Limit GPU memory oversubscription: set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and optionally `XLA_PYTHON_CLIENT_MEM_FRACTION` per job. Prefer chunking (`chunk_size`, `batch_size`) over spilling.

## Compilation and caching
- JIT everything in forward paths; avoid recompiles by:
  - Keeping static shapes: pad/pack trajectories to consistent length per job when possible.
  - Passing config booleans as static args (`static_argnames`) only when they truly change infrequently.
  - Reusing meshes and batch dimensions across calls (same `chunk_size`, `batch_size`, `n_q`).
- Warm up once per executable (one dummy call) before timing/production runs; reuse processes to keep XLA cache hot.

## Sharding/distribution
- Single host, multi-GPU: use `pmap`/`pjit` over trajectory/batch dimension. Ensure `OutputIndex` and feature arrays are replicated; coordinates are sharded. Keep `chunk_size` ≥ 256 to amortize collective overhead.
- Multi-host (SLURM/MPI): one process per GPU; use `jax.distributed.initialize` with SLURM env (`SLURM_NODEID`, `SLURM_PROCID`). Pass identical feature npz to all ranks; only coordinates are sharded/streamed.
- Avoid host collectives in hot loops; aggregate outputs via `all_gather` on the batch axis, then write once on rank 0.

## Data movement
- Mostly likely to featurise on CPU, write `{prefix}_features.npz` + topology JSON. Transfer to GPU once per job and keep resident.
- Stream trajectories in batches: host → device using pinned arrays; prefer `batch_size` small enough to fit device memory with six-partials (SAXS) and contact masks (HDX).
- For static single-structure runs, keep coordinates on device across hydration grid search / c1-c2 sweeps to avoid retransfer.

## Failure/edge handling
- Detect low-device count (`jax.local_device_count() == 1`) and fall back to single-device execution without sharding.
- If HETATM-heavy systems blow memory, reduce `chunk_size` first, then `batch_size`.
- For deterministic gradients: fix PRNG seeds for any stochastic preprocessing; no stochasticity is present in forward kernels.

## Future hook
- Differentiable SASA will replace static `solvent_acc`; revisit sharding and memory once SASA kernels are introduced.
