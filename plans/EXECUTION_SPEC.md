# BioFeaturisers Execution Spec (v0.1)

## 1) Objective

Define the implementation outcome for the current phase in one sentence.

Example:
Deliver production-ready HDX and SAXS forward pipelines with verified core numerics, tests, and reproducible execution settings.

## 2) Scope

In scope:
- Core numerics needed by active modules
- Module implementation tasks mapped to plan docs
- Verification commands and done artifacts

Out of scope:
- New research directions not already captured in `plans/*.md`
- Unscoped refactors unrelated to active Spec IDs

## 3) Plan source mapping rule

Every work item must map to one or more plan sources (file + section) from:

- `plans/architecture.md`
- `plans/pairwise.md`
- `plans/HDX.md`
- `plans/SAXS.md`
- `plans/testing.md`
- `plans/devices_execution.md`
- `plans/internal_benchmarking.md`

If an item cannot be mapped, it must not be started.

## 4) Work items (authoritative)

| Spec ID | Requirement | Plan Sources | Files to Change | Verify Command(s) | Done Artifact(s) | Depends On |
|---|---|---|---|---|---|---|
| CORE-001 | Implement safe math primitives (`safe_sqrt`, `safe_sqrt_sym`, `safe_sinc`) | `pairwise.md`, `testing.md` | `core/safe_math.py` | `pytest -k safe_math` | Passing safe_math tests | - |
| CORE-002 | Implement pairwise kernels (`dist_matrix_asymmetric`, `dist_matrix_block`, `dist_from_sq_block`) | `pairwise.md`, `architecture.md` | `core/pairwise.py` | `pytest -k pairwise` | Passing pairwise tests | CORE-001 |
| HDX-001 | Implement HDX featurise/forward/predict path | `HDX.md`, `architecture.md` | `hdx/featurise.py`, `hdx/forward.py`, `hdx/predict.py` | `pytest -k hdx` | Passing HDX tests | CORE-002 |
| SAXS-001 | Implement SAXS featurise/forward/predict path | `SAXS.md`, `architecture.md` | `saxs/featurise.py`, `saxs/forward.py`, `saxs/predict.py` | `pytest -k saxs` | Passing SAXS tests | CORE-002 |
| TEST-001 | Implement regression and gradient checks | `testing.md` | `tests/**/*` | `pytest` | Full test pass | HDX-001, SAXS-001 |
| DEV-001 | Add device/distributed execution hooks | `devices_execution.md` | runtime/config execution files | project test command | Verified runtime settings | TEST-001 |
| BENCH-001 | Add benchmark harness and reporting | `internal_benchmarking.md` | benchmark harness files | benchmark command set | Benchmark summary tables | DEV-001 |

## 5) Gates

- No implementation before reading mapped plan sources.
- WIP limit is 1 (`in_progress` count must be exactly 0 or 1).
- A Spec ID can start only when all dependencies are `done`.
- No `done` status without passing verify commands.
- Blocked work must include a concrete unblock condition.

## 6) Execution protocol

1. Select next ready Spec ID (all dependencies done).
2. Set status to `in_progress`.
3. Implement only files listed in that Spec ID.
4. Run listed verify commands.
5. Mark `done` if verification passes; else keep `in_progress` or mark `blocked` with reason.
6. Publish checkpoint and move to next ready item.

## 7) SQL status model

Use `todos` and `todo_deps` with IDs aligned to Spec IDs where practical.

Status values:
- `pending`
- `in_progress`
- `done`
- `blocked`

## 8) Checkpoint format (required after each item)

- Completed: `<Spec ID> - <title>`
- Changed: `<key files>`
- Verified: `<commands and pass/fail>`
- Next ready: `<Spec ID>`
- Risks/blocks: `<short note>`

## 9) Change control

Any scope change must update:

1. This spec (`Scope` and `Work items`)
2. SQL todos/dependencies
3. Verification expectations for affected Spec IDs
