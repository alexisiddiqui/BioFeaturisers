# Internal benchmarking plan

## Objectives
- Quantify correctness, throughput, latency, and memory for HDX and SAXS across mixed CPU/GPU clusters.
- Validate scale limits and tuning levers (`chunk_size`, `batch_size`, hydration grid search) before external releases.
- Produce reproducible numbers for changelogs and regressions.

## Scenarios
- HDX single structure: 1–5k atoms; measure ln_Pf, Nc/Nh correctness vs reference.
- HDX trajectories: 100–1,000 frames of 5k–20k atoms; test batching/sharding.
- SAXS single structure: 5k–50k atoms, n_q=300; six-partial path stress.
- SAXS grid search: c1/c2 sweep with cached six-partials; verify reuse wins.
- I/O: feature save/load throughput; host→device transfer cost for features + coords.

## Metrics to record
- End-to-end latency and throughput (structures/s or frames/s).
- Peak device memory (from backend profiler) and host memory.
- Compile times per executable and cache hit ratio on reruns.
- Numerical drift: max/mean abs/rel error vs reference (CPU fp64 baseline with numpy).
- Determinism: run-to-run stddev for outputs on identical seeds/configs.

## Hardware matrix
- CPU: 8c and 32c nodes (AVX2/AVX-512); note x64 vs x32 performance.
- GPU: A100 40/80GB, V100 16/32GB if available

## Methodology
- Use synthetic compact proteins (random but fixed seeds) to avoid I/O variability; also include one real PDB for plausibility.
- Warm up once to compile; discard first run in timing. Reuse same process to keep XLA cache hot.
- Fix seeds for any stochastic preprocessing (none in forward); pin versions of JAX/biotite.
- Keep shapes constant per run to avoid recompiles; log `chunk_size`, `batch_size`, `n_q`, device count.
- Measure with/without sharding (`pmap`/`pjit`) to show scaling; focus on single-device baselines.
- Record command lines and env (`XLA_*`, `JAX_*` flags); store raw logs + summaries in `/benchmarks/` (not committed if large).

## Reporting
- For each scenario/hardware, capture: config, mean/median latency (±std), throughput, peak mem, compile time, correctness deltas.
- Summaries should be table-first; include brief tuning notes (e.g., “chunk_size 256 → -20% mem, +5% time”).
- Flag regressions >5% runtime or memory for follow-up tests.
