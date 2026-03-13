# BioFeaturisers — Strict Implementation Workflow

## Why this exists

The project has strong technical plans (`architecture.md`, `pairwise.md`, `HDX.md`, `SAXS.md`, `testing.md`, `devices_execution.md`, `internal_benchmarking.md`), but execution can drift when TODOs are created without enforcing plan coverage.

This workflow converts plans into a deterministic execution contract so work is fully implemented, verified, and dependency-safe.

---

## Non-negotiable rules

1. **No code before plan read**
   - Before any implementation TODO starts, all relevant plan docs must be read and summarized.

2. **No TODO without source mapping**
   - Every TODO must cite at least one source plan document and section.

3. **WIP limit = 1**
   - Only one TODO can be `in_progress` at a time.

4. **Dependency lock**
   - A TODO may start only when all dependencies are `done`.

5. **No done without proof**
   - A TODO is `done` only when its verification commands pass and expected files exist.

6. **No silent carryover**
   - If a TODO is incomplete, it remains `in_progress` or `blocked` with a concrete reason.

---

## Canonical implementation order for this repo

Use this top-level dependency graph:

1. `architecture.md` (system contract)
2. `pairwise.md` (shared numerical primitives)
3. `HDX.md` and `SAXS.md` (module implementations)
4. `testing.md` (regression + module validation)
5. `devices_execution.md` (runtime/distributed behavior)
6. `internal_benchmarking.md` (performance and reproducibility)

Practical rule: do not begin module-level optimization before core pairwise/safe math and tests are in place.

---

## TODO schema (required fields)

Each TODO must include:

- `id`: stable kebab-case identifier
- `title`: single clear deliverable
- `description`: exact files + behavior to implement
- `plan_sources`: plan docs/sections it implements
- `verify_commands`: commands that must pass
- `done_artifacts`: files or outputs expected at completion

If any of these are missing, the TODO is invalid and must not be started.

---

## SQL orchestration contract

Use `todos` + `todo_deps` as the execution engine.

### 1) Create mapped TODOs

```sql
INSERT INTO todos (id, title, description, status) VALUES
  ('core-safe-math', 'Implement core safe math primitives', 'Implement/validate safe_sqrt, safe_sqrt_sym, safe_sinc custom VJP in core per pairwise/testing plans.', 'pending'),
  ('core-pairwise', 'Implement pairwise distance kernels', 'Implement dist_matrix_asymmetric, dist_matrix_block, dist_from_sq_block and chunked apply patterns.', 'pending'),
  ('hdx-module', 'Implement HDX featurise/forward/predict', 'Implement HDX config/features/forward using BV + Wan sigmoid with exclusion masks and optional HDXrate handling.', 'pending'),
  ('saxs-module', 'Implement SAXS featurise/forward/predict', 'Implement FoXS-style six partial sums, safe sinc usage, c1/c2 recombination and predict path.', 'pending'),
  ('test-suite', 'Implement module/regression tests', 'Add tests from testing plan for core, HDX, SAXS, gradient checks and shape invariants.', 'pending'),
  ('devices-runtime', 'Implement device/distributed runtime hooks', 'Add execution controls for device policy, batching, chunking, and distributed-safe behavior.', 'pending'),
  ('benchmarking', 'Implement internal benchmark harness', 'Add benchmark scenarios and report tables for correctness/perf/memory regressions.', 'pending');
```

### 2) Declare dependencies

```sql
INSERT INTO todo_deps (todo_id, depends_on) VALUES
  ('core-pairwise', 'core-safe-math'),
  ('hdx-module', 'core-pairwise'),
  ('saxs-module', 'core-pairwise'),
  ('test-suite', 'hdx-module'),
  ('test-suite', 'saxs-module'),
  ('devices-runtime', 'test-suite'),
  ('benchmarking', 'devices-runtime');
```

### 3) Pull the next ready TODO

```sql
SELECT t.*
FROM todos t
WHERE t.status = 'pending'
  AND NOT EXISTS (
    SELECT 1
    FROM todo_deps d
    JOIN todos dep ON dep.id = d.depends_on
    WHERE d.todo_id = t.id
      AND dep.status != 'done'
  )
ORDER BY t.created_at
LIMIT 1;
```

### 4) Status transitions (strict)

```sql
UPDATE todos SET status = 'in_progress' WHERE id = :todo_id;
-- implement + verify
UPDATE todos SET status = 'done' WHERE id = :todo_id;
```

On failure or ambiguity:

```sql
UPDATE todos
SET status = 'blocked',
    description = description || ' | BLOCKED: <reason>'
WHERE id = :todo_id;
```

---

## Execution loop (must be followed every cycle)

1. Read relevant plan docs and write a brief implementation map (what will be implemented now).
2. Query one ready TODO.
3. Set it to `in_progress`.
4. Implement only the scoped files/behavior for that TODO.
5. Run verification commands defined for that TODO.
6. If all pass, mark `done`; otherwise keep `in_progress` or mark `blocked` with reason.
7. Log a short completion note: what changed, what passed, what remains.
8. Repeat from step 2.

This loop prevents partial execution and prevents skipping plan requirements.

---

## Definition of Done (DoD) gate

A TODO can be marked `done` only if all are true:

- Plan requirements mapped in the TODO description are implemented.
- Target files were changed as expected.
- Verification commands passed.
- Related docs/configs updated where required.
- No unresolved blocker remains for the TODO scope.

If any condition fails, it is not done.

---

## Agent prompt contract (recommended)

Use this at the start of each implementation session:

> Read `plans/architecture.md`, `plans/pairwise.md`, `plans/HDX.md`, `plans/SAXS.md`, `plans/testing.md`, `plans/devices_execution.md`, and `plans/internal_benchmarking.md`.  
> Build a dependency-aware TODO graph in SQL (`todos`, `todo_deps`) with explicit file scopes and verification commands.  
> Execute one ready TODO at a time (WIP=1), always updating status transitions (`pending -> in_progress -> done/blocked`).  
> Do not mark a TODO done without passing verification commands.  
> After each TODO, output a concise checkpoint (changes, tests, next ready TODO).

---

## Common failure modes this workflow prevents

- TODOs created without reading plans
- Parallel partially-complete implementation branches
- “Done” states without tests/verification
- Starting advanced optimization before core numerics are stable
- Losing execution context between sessions

---

## Minimal checkpoint format

After each TODO completion, emit:

- `Completed:` TODO id/title
- `Changed:` key files
- `Verified:` commands and pass/fail
- `Next ready:` TODO id/title
- `Risks/blocks:` short note (if any)

This keeps execution auditable and resumable.
