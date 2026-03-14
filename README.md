# BioFeaturisers

HDX, SAXS and more implemented in JAX (and eventually PyTorch).

## CLI quickstart

The Typer CLI exposes:

- `biofeaturisers hdx featurise|forward|predict`
- `biofeaturisers saxs featurise|forward|predict`

Examples:

```bash
biofeaturisers hdx featurise \
  --structure tests/fixtures/1ubq_A_1_15.pdb \
  --out /tmp/ubq_hdx \
  --env device=cpu,disable_preallocation=true

biofeaturisers hdx forward \
  --features /tmp/ubq_hdx \
  --coords /tmp/coords.npy \
  --out /tmp/ubq_hdx_run \
  --env device=cpu

biofeaturisers saxs predict \
  --structure tests/fixtures/1ubq_A_1_15.pdb \
  --out /tmp/ubq_saxs_run \
  --config /tmp/saxs.toml \
  --env /tmp/env.toml
```

`--env` supports either:

- Inline key/value pairs (`device=cpu,gpu_index=0,disable_preallocation=true`)
- A TOML file path (top-level keys or `[compute]` section)

## I/O artifacts

Feature featurisation persists:

- `{prefix}_features.npz`
- `{prefix}_topology.json`

Forward/predict output persistence:

- HDX: `{prefix}_hdx_output.npz` + `{prefix}_hdx_index.json`
- SAXS: `{prefix}_saxs_output.npz` + `{prefix}_saxs_index.json`

Experimental format readers are available in `biofeaturisers.io.formats`:

- SAXS: `.dat`, `.fit`, `.csv`
- HDX: `.csv`

## Future Work

- [ ] Research dSASA - need to check this

