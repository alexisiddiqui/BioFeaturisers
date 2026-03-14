"""Typer CLI for BioFeaturisers feature generation and forward workflows."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import typer

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility.
    import tomli as tomllib

from biofeaturisers.config import HDXConfig, SAXSConfig
from biofeaturisers.env import Backend, ComputeConfig, Device
from biofeaturisers.io.save import save_hdx_output, save_saxs_output

app = typer.Typer(help="BioFeaturisers command-line interface.")
hdx_app = typer.Typer(help="HDX featurise/forward/predict commands.")
saxs_app = typer.Typer(help="SAXS featurise/forward/predict commands.")
app.add_typer(hdx_app, name="hdx")
app.add_typer(saxs_app, name="saxs")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"TOML document at {path} must be a table/object")
    return payload


def _load_dataclass_config(
    path: Path | None, config_cls: type[HDXConfig] | type[SAXSConfig], section_name: str
) -> HDXConfig | SAXSConfig:
    if path is None:
        return config_cls()

    payload = _load_toml(path)
    section = payload.get(section_name)
    if isinstance(section, dict):
        config_payload = dict(section)
    else:
        config_payload = dict(payload)

    allowed = {field.name for field in fields(config_cls)}
    unknown = sorted(set(config_payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {config_cls.__name__} config fields: {joined}")
    return config_cls(**config_payload)


def _parse_inline_mapping(spec: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    entries = [entry.strip() for entry in spec.split(",") if entry.strip()]
    if not entries:
        raise ValueError("Inline env spec is empty")
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid inline env item '{entry}'; expected key=value")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if key == "":
            raise ValueError(f"Invalid inline env item '{entry}'; key cannot be empty")
        mapping[key] = value
    return mapping


def _coerce_compute_config(mapping: dict[str, Any]) -> ComputeConfig:
    allowed = {field.name for field in fields(ComputeConfig)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown ComputeConfig fields: {joined}")

    kwargs: dict[str, Any] = {}
    if "backend" in mapping:
        kwargs["backend"] = Backend(str(mapping["backend"]).strip().lower())
    if "device" in mapping:
        kwargs["device"] = Device(str(mapping["device"]).strip().lower())
    if "gpu_index" in mapping:
        kwargs["gpu_index"] = int(mapping["gpu_index"])
    if "disable_preallocation" in mapping:
        kwargs["disable_preallocation"] = _parse_bool(mapping["disable_preallocation"])
    if "enable_x64" in mapping:
        kwargs["enable_x64"] = _parse_bool(mapping["enable_x64"])
    return ComputeConfig(**kwargs)


def _apply_env(env: str | None) -> ComputeConfig | None:
    if env is None:
        return None

    if "=" in env:
        mapping = _parse_inline_mapping(env)
    else:
        env_path = Path(env)
        payload = _load_toml(env_path)
        compute_section = payload.get("compute")
        mapping = dict(compute_section) if isinstance(compute_section, dict) else dict(payload)

    cfg = _coerce_compute_config(mapping)
    cfg.configure()
    return cfg


def _load_structure(structure_path: Path):
    suffix = structure_path.suffix.lower()
    if suffix != ".pdb":
        raise ValueError(f"Unsupported structure format '{suffix}'. Only .pdb is supported.")
    from biotite.structure.io.pdb import PDBFile

    pdb = PDBFile.read(structure_path)
    return pdb.get_structure(model=1)


def _load_array(path: Path, *, preferred_key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if preferred_key is not None and preferred_key in data.files:
                key = preferred_key
            elif "coords" in data.files:
                key = "coords"
            elif len(data.files) == 1:
                key = data.files[0]
            else:
                joined = ", ".join(data.files)
                raise ValueError(
                    f"{path} has multiple arrays ({joined}); provide a file with a 'coords' key."
                )
            return np.asarray(data[key], dtype=np.float32)
    raise ValueError(f"Unsupported array format '{suffix}'. Use .npy or .npz files.")


def _load_weights(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    weights = _load_array(path, preferred_key="weights")
    if weights.ndim != 1:
        raise ValueError(f"weights must be rank-1, got shape {weights.shape}")
    return weights


def _run_hdx_forward(
    coords: np.ndarray,
    *,
    features,
    config: HDXConfig,
    weights: np.ndarray | None,
) -> dict[str, Any]:
    import jax.numpy as jnp

    from biofeaturisers.core.ensemble import apply_forward
    from biofeaturisers.hdx.forward import hdx_forward

    coords_arr = jnp.asarray(coords, dtype=jnp.float32)
    if coords_arr.ndim == 2:
        return hdx_forward(coords_arr, features=features, config=config)
    if coords_arr.ndim != 3:
        raise ValueError(
            f"coords must have shape (n_atoms, 3) or (n_frames, n_atoms, 3), got {coords_arr.shape}"
        )

    def _stack(frame):
        result = hdx_forward(frame, features=features, config=config)
        return jnp.stack((result["Nc"], result["Nh"], result["ln_Pf"]), axis=0)

    weights_arr = None if weights is None else jnp.asarray(weights, dtype=jnp.float32)
    stacked = apply_forward(
        _stack,
        coords=coords_arr,
        weights=weights_arr,
        batch_size=config.batch_size,
    )
    return {"Nc": stacked[0], "Nh": stacked[1], "ln_Pf": stacked[2]}


def _hdx_index_for_probe_outputs(features) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    can_exchange = np.asarray(features.can_exchange, dtype=bool)
    res_keys = np.asarray(features.res_keys, dtype=str)
    res_names = np.asarray(features.res_names, dtype=str)
    if res_keys.shape != can_exchange.shape:
        raise ValueError("HDX feature metadata mismatch: res_keys and can_exchange shapes differ")
    if res_names.shape != can_exchange.shape:
        raise ValueError("HDX feature metadata mismatch: res_names and can_exchange shapes differ")

    res_keys_probe = res_keys[can_exchange]
    res_names_probe = res_names[can_exchange]
    can_exchange_probe = np.ones(res_keys_probe.shape[0], dtype=bool)
    if features.kint is None:
        kint_probe = None
    else:
        kint_probe = np.asarray(features.kint, dtype=np.float32)[can_exchange]
    return res_keys_probe, res_names_probe, can_exchange_probe, kint_probe


@hdx_app.command("featurise")
def hdx_featurise_cmd(
    structure: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    out: str = typer.Option(..., help="Output prefix (writes *_features.npz + *_topology.json)."),
    config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True, help="Optional HDX TOML config."
    ),
    env: str | None = typer.Option(
        None, help="ComputeConfig as inline key=value pairs or TOML file path."
    ),
) -> None:
    _apply_env(env)
    cfg = _load_dataclass_config(config, HDXConfig, "hdx")
    atom_array = _load_structure(structure)

    from biofeaturisers.hdx.featurise import featurise

    features = featurise(atom_array=atom_array, config=cfg)
    features.save(out)
    typer.echo(f"Saved HDX features: {out}_features.npz + {out}_topology.json")


@hdx_app.command("forward")
def hdx_forward_cmd(
    features: str = typer.Option(..., help="Feature prefix (without *_features.npz suffix)."),
    coords: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    out: str = typer.Option(..., help="Output prefix for *_hdx_output.npz and *_hdx_index.json."),
    config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True, help="Optional HDX TOML config."
    ),
    weights: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional weights array (.npy/.npz) for trajectory inputs.",
    ),
    env: str | None = typer.Option(
        None, help="ComputeConfig as inline key=value pairs or TOML file path."
    ),
) -> None:
    _apply_env(env)
    cfg = _load_dataclass_config(config, HDXConfig, "hdx")
    coords_arr = _load_array(coords)
    weights_arr = _load_weights(weights)

    from biofeaturisers.hdx.features import HDXFeatures

    feature_obj = HDXFeatures.load(features)
    result = _run_hdx_forward(
        coords_arr,
        features=feature_obj,
        config=cfg,
        weights=weights_arr,
    )
    res_keys_probe, res_names_probe, can_exchange_probe, kint_probe = _hdx_index_for_probe_outputs(
        feature_obj
    )
    save_hdx_output(
        out,
        nc=np.asarray(result["Nc"], dtype=np.float32),
        nh=np.asarray(result["Nh"], dtype=np.float32),
        ln_pf=np.asarray(result["ln_Pf"], dtype=np.float32),
        res_keys=res_keys_probe,
        res_names=res_names_probe,
        can_exchange=can_exchange_probe,
        kint=kint_probe,
    )
    typer.echo(f"Saved HDX outputs: {out}_hdx_output.npz + {out}_hdx_index.json")


@hdx_app.command("predict")
def hdx_predict_cmd(
    structure: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    out: str = typer.Option(..., help="Output prefix for *_hdx_output.npz and *_hdx_index.json."),
    coords: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True, help="Optional coords (.npy/.npz)."
    ),
    config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True, help="Optional HDX TOML config."
    ),
    weights: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional weights array (.npy/.npz) for trajectory inputs.",
    ),
    env: str | None = typer.Option(
        None, help="ComputeConfig as inline key=value pairs or TOML file path."
    ),
) -> None:
    _apply_env(env)
    cfg = _load_dataclass_config(config, HDXConfig, "hdx")
    atom_array = _load_structure(structure)
    coords_arr = None if coords is None else _load_array(coords)
    weights_arr = _load_weights(weights)

    from biofeaturisers.hdx.featurise import featurise
    from biofeaturisers.hdx.predict import predict

    result = predict(atom_array=atom_array, config=cfg, coords=coords_arr, weights=weights_arr)
    feature_obj = featurise(atom_array=atom_array, config=cfg)
    res_keys_probe, res_names_probe, can_exchange_probe, kint_probe = _hdx_index_for_probe_outputs(
        feature_obj
    )
    save_hdx_output(
        out,
        nc=np.asarray(result["Nc"], dtype=np.float32),
        nh=np.asarray(result["Nh"], dtype=np.float32),
        ln_pf=np.asarray(result["ln_Pf"], dtype=np.float32),
        uptake_curves=(
            np.asarray(result["uptake"], dtype=np.float32) if "uptake" in result else None
        ),
        res_keys=res_keys_probe,
        res_names=res_names_probe,
        can_exchange=can_exchange_probe,
        kint=kint_probe,
    )
    typer.echo(f"Saved HDX outputs: {out}_hdx_output.npz + {out}_hdx_index.json")


@saxs_app.command("featurise")
def saxs_featurise_cmd(
    structure: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    out: str = typer.Option(..., help="Output prefix (writes *_features.npz + *_topology.json)."),
    config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True, help="Optional SAXS TOML config."
    ),
    env: str | None = typer.Option(
        None, help="ComputeConfig as inline key=value pairs or TOML file path."
    ),
) -> None:
    _apply_env(env)
    cfg = _load_dataclass_config(config, SAXSConfig, "saxs")
    atom_array = _load_structure(structure)

    from biofeaturisers.saxs.featurise import featurise

    features = featurise(atom_array=atom_array, config=cfg)
    features.save(out)
    typer.echo(f"Saved SAXS features: {out}_features.npz + {out}_topology.json")


@saxs_app.command("forward")
def saxs_forward_cmd(
    features: str = typer.Option(..., help="Feature prefix (without *_features.npz suffix)."),
    coords: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    out: str = typer.Option(..., help="Output prefix for *_saxs_output.npz and *_saxs_index.json."),
    config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True, help="Optional SAXS TOML config."
    ),
    weights: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional weights array (.npy/.npz) for trajectory inputs.",
    ),
    env: str | None = typer.Option(
        None, help="ComputeConfig as inline key=value pairs or TOML file path."
    ),
) -> None:
    _apply_env(env)
    cfg = _load_dataclass_config(config, SAXSConfig, "saxs")
    coords_arr = _load_array(coords)
    weights_arr = _load_weights(weights)

    from biofeaturisers.saxs.features import SAXSFeatures
    from biofeaturisers.saxs.forward import forward

    feature_obj = SAXSFeatures.load(features)
    i_q = forward(coords=coords_arr, features=feature_obj, config=cfg, weights=weights_arr)
    save_saxs_output(
        out,
        i_q=np.asarray(i_q, dtype=np.float32),
        q_values=np.asarray(feature_obj.q_values, dtype=np.float32),
        chain_ids=np.asarray(feature_obj.chain_ids, dtype=str),
        c1_used=float(cfg.c1),
        c2_used=float(cfg.c2),
    )
    typer.echo(f"Saved SAXS outputs: {out}_saxs_output.npz + {out}_saxs_index.json")


@saxs_app.command("predict")
def saxs_predict_cmd(
    structure: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    out: str = typer.Option(..., help="Output prefix for *_saxs_output.npz and *_saxs_index.json."),
    coords: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True, help="Optional coords (.npy/.npz)."
    ),
    config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True, help="Optional SAXS TOML config."
    ),
    weights: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional weights array (.npy/.npz) for trajectory inputs.",
    ),
    env: str | None = typer.Option(
        None, help="ComputeConfig as inline key=value pairs or TOML file path."
    ),
) -> None:
    _apply_env(env)
    cfg = _load_dataclass_config(config, SAXSConfig, "saxs")
    atom_array = _load_structure(structure)
    coords_arr = None if coords is None else _load_array(coords)
    weights_arr = _load_weights(weights)

    from biofeaturisers.saxs.featurise import featurise
    from biofeaturisers.saxs.predict import predict

    prediction = predict(atom_array=atom_array, config=cfg, coords=coords_arr, weights=weights_arr)
    feature_obj = featurise(atom_array=atom_array, config=cfg)

    if isinstance(prediction, tuple):
        i_q, _chi2, c1_used, c2_used = prediction
    else:
        i_q = prediction
        c1_used = float(cfg.c1)
        c2_used = float(cfg.c2)

    save_saxs_output(
        out,
        i_q=np.asarray(i_q, dtype=np.float32),
        q_values=np.asarray(feature_obj.q_values, dtype=np.float32),
        chain_ids=np.asarray(feature_obj.chain_ids, dtype=str),
        c1_used=float(c1_used),
        c2_used=float(c2_used),
    )
    typer.echo(f"Saved SAXS outputs: {out}_saxs_output.npz + {out}_saxs_index.json")


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()
