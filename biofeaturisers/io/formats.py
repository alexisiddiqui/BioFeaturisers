"""Parsers for experimental SAXS/HDX files and output index JSON payloads."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_NAME_NORMALIZER = re.compile(r"[^a-z0-9]+")


def _normalize_name(name: str) -> str:
    return _NAME_NORMALIZER.sub("", name.lower())


def _read_numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            tokens = line.replace(",", " ").split()
            values: list[float] = []
            for token in tokens:
                try:
                    values.append(float(token))
                except ValueError:
                    values = []
                    break
            if len(values) >= 2:
                rows.append(values)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return rows


def _saxs_from_numeric_rows(rows: list[list[float]]) -> "SAXSData":
    q = np.asarray([row[0] for row in rows], dtype=np.float32)
    intensity = np.asarray([row[1] for row in rows], dtype=np.float32)
    sigma = (
        np.asarray([row[2] for row in rows], dtype=np.float32)
        if all(len(row) >= 3 for row in rows)
        else None
    )
    return SAXSData(q=q, intensity=intensity, sigma=sigma)


def _select_field(
    field_map: dict[str, str], candidates: tuple[str, ...], *, required: bool = True
) -> str | None:
    for name in candidates:
        if name in field_map:
            return field_map[name]
    if required:
        raise ValueError(f"Missing required field; expected one of {candidates}")
    return None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(slots=True)
class SAXSData:
    """Parsed SAXS dataset with optional per-point uncertainty."""

    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=np.float32)
        self.intensity = np.asarray(self.intensity, dtype=np.float32)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=np.float32)

        if self.q.ndim != 1 or self.intensity.ndim != 1:
            raise ValueError("q and intensity must be rank-1")
        if self.q.shape != self.intensity.shape:
            raise ValueError("q and intensity must have matching lengths")
        if self.sigma is not None and self.sigma.shape != self.q.shape:
            raise ValueError("sigma must match q/intensity shape when provided")


@dataclass(slots=True)
class HDXData:
    """Parsed HDX-MS uptake table."""

    peptide: np.ndarray
    start: np.ndarray
    end: np.ndarray
    timepoint: np.ndarray
    deuterium: np.ndarray

    def __post_init__(self) -> None:
        self.peptide = np.asarray(self.peptide, dtype=str)
        self.start = np.asarray(self.start, dtype=np.int32)
        self.end = np.asarray(self.end, dtype=np.int32)
        self.timepoint = np.asarray(self.timepoint, dtype=np.float32)
        self.deuterium = np.asarray(self.deuterium, dtype=np.float32)

        if self.peptide.ndim != 1:
            raise ValueError("peptide must be rank-1")
        n_rows = int(self.peptide.shape[0])
        if int(self.start.shape[0]) != n_rows:
            raise ValueError("start length must match peptide length")
        if int(self.end.shape[0]) != n_rows:
            raise ValueError("end length must match peptide length")
        if int(self.timepoint.shape[0]) != n_rows:
            raise ValueError("timepoint length must match peptide length")
        if int(self.deuterium.shape[0]) != n_rows:
            raise ValueError("deuterium length must match peptide length")


def load_saxs_data(path: str | Path) -> SAXSData:
    """Load SAXS data from `.dat`, `.fit`, or `.csv` inputs."""
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix in {".dat", ".fit"}:
        return _saxs_from_numeric_rows(_read_numeric_rows(path_obj))
    if suffix == ".csv":
        return load_saxs_csv(path_obj)
    raise ValueError(f"Unsupported SAXS file extension '{suffix}' for {path_obj}")


def load_saxs_csv(path: str | Path) -> SAXSData:
    """Load SAXS data from CSV with normalized column matching."""
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames:
            field_map = {_normalize_name(name): name for name in fieldnames if name}
            q_field = _select_field(
                field_map, ("q", "qvalue", "qvalues", "qa1", "qangstrominverse")
            )
            i_field = _select_field(
                field_map, ("i", "iq", "intensity", "iexp", "iexperimental")
            )
            sigma_field = _select_field(
                field_map, ("sigma", "error", "err", "uncertainty", "std"), required=False
            )

            q_vals: list[float] = []
            i_vals: list[float] = []
            sigma_vals: list[float] = []
            for row in reader:
                if row is None:
                    continue
                if row.get(q_field, "").strip() == "" and row.get(i_field, "").strip() == "":
                    continue
                q_vals.append(float(row[q_field]))
                i_vals.append(float(row[i_field]))
                if sigma_field is not None:
                    sigma_vals.append(float(row[sigma_field]))

            sigma_arr = np.asarray(sigma_vals, dtype=np.float32) if sigma_field is not None else None
            return SAXSData(
                q=np.asarray(q_vals, dtype=np.float32),
                intensity=np.asarray(i_vals, dtype=np.float32),
                sigma=sigma_arr,
            )

    return _saxs_from_numeric_rows(_read_numeric_rows(path_obj))


def load_hdx_csv(path: str | Path) -> HDXData:
    """Load HDX uptake table from CSV."""
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"HDX CSV {path_obj} must include a header row")

        field_map = {_normalize_name(name): name for name in reader.fieldnames if name}
        peptide_field = _select_field(field_map, ("peptide", "sequence", "peptidesequence"))
        start_field = _select_field(field_map, ("start", "startresidue", "startres"))
        end_field = _select_field(field_map, ("end", "endresidue", "endres"))
        time_field = _select_field(field_map, ("timepoint", "time", "times", "time_s", "seconds"))
        deut_field = _select_field(field_map, ("deuterium", "uptake", "uptakeda", "d"))

        peptide: list[str] = []
        start: list[int] = []
        end: list[int] = []
        timepoint: list[float] = []
        deuterium: list[float] = []

        for row in reader:
            if row is None:
                continue
            if row.get(peptide_field, "").strip() == "":
                continue
            peptide.append(str(row[peptide_field]).strip())
            start.append(int(float(row[start_field])))
            end.append(int(float(row[end_field])))
            timepoint.append(float(row[time_field]))
            deuterium.append(float(row[deut_field]))

    return HDXData(
        peptide=np.asarray(peptide, dtype=str),
        start=np.asarray(start, dtype=np.int32),
        end=np.asarray(end, dtype=np.int32),
        timepoint=np.asarray(timepoint, dtype=np.float32),
        deuterium=np.asarray(deuterium, dtype=np.float32),
    )


def parse_hdx_index_payload(payload: Any) -> dict[str, np.ndarray | None]:
    """Normalize HDX index payloads from canonical or custom JSON shapes."""
    if isinstance(payload, dict) and "res_keys" in payload:
        res_keys = np.asarray(payload["res_keys"], dtype=str)
        res_names = np.asarray(payload.get("res_names", ["UNK"] * len(res_keys)), dtype=str)
        can_exchange = np.asarray(payload.get("can_exchange", [True] * len(res_keys)), dtype=bool)
        kint_value = payload.get("kint")
        kint = None if kint_value is None else np.asarray(kint_value, dtype=np.float32)
        return {
            "res_keys": res_keys,
            "res_names": res_names,
            "can_exchange": can_exchange,
            "kint": kint,
        }

    records: list[dict[str, Any]]
    if isinstance(payload, list):
        records = [dict(item) for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict) and all(isinstance(value, dict) for value in payload.values()):
        records = [
            {"res_key": str(key), **dict(value)}
            for key, value in payload.items()
            if isinstance(value, dict)
        ]
    elif isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        records = [dict(item) for item in payload["entries"] if isinstance(item, dict)]
    else:
        raise ValueError("Unsupported HDX index JSON representation")

    res_keys: list[str] = []
    res_names: list[str] = []
    can_exchange: list[bool] = []
    kint_values: list[float] = []
    has_kint = False
    for record in records:
        key = str(
            record.get("res_key")
            or record.get("residue_key")
            or record.get("key")
            or record.get("resid")
            or ""
        )
        if key == "":
            raise ValueError("HDX index record is missing residue key")
        res_keys.append(key)
        res_names.append(str(record.get("res_name") or record.get("name") or "UNK"))
        can_exchange.append(bool(record.get("can_exchange", record.get("exchangeable", True))))
        value = record.get("kint")
        if value is None:
            kint_values.append(np.nan)
        else:
            has_kint = True
            kint_values.append(float(value))

    return {
        "res_keys": np.asarray(res_keys, dtype=str),
        "res_names": np.asarray(res_names, dtype=str),
        "can_exchange": np.asarray(can_exchange, dtype=bool),
        "kint": np.asarray(kint_values, dtype=np.float32) if has_kint else None,
    }


def parse_saxs_index_payload(payload: Any) -> dict[str, Any]:
    """Normalize SAXS index payloads from canonical or custom JSON shapes."""
    if isinstance(payload, dict) and "chain_ids" in payload:
        chain_ids = np.asarray(payload.get("chain_ids", []), dtype=str)
        atom_counts_raw = payload.get("atom_counts", {})
        atom_counts = {str(key): int(value) for key, value in dict(atom_counts_raw).items()}
        c1 = payload.get("c1_used", payload.get("c1"))
        c2 = payload.get("c2_used", payload.get("c2"))
        return {
            "chain_ids": chain_ids,
            "atom_counts": atom_counts,
            "c1_used": None if c1 is None else float(c1),
            "c2_used": None if c2 is None else float(c2),
        }

    chain_records: list[dict[str, Any]]
    container: dict[str, Any] = {}
    if isinstance(payload, list):
        chain_records = [dict(item) for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("chains"), list):
        chain_records = [dict(item) for item in payload["chains"] if isinstance(item, dict)]
        container = payload
    elif isinstance(payload, dict) and all(isinstance(value, (int, float)) for value in payload.values()):
        atom_counts = {str(key): int(value) for key, value in payload.items()}
        return {
            "chain_ids": np.asarray(list(atom_counts.keys()), dtype=str),
            "atom_counts": atom_counts,
            "c1_used": None,
            "c2_used": None,
        }
    else:
        raise ValueError("Unsupported SAXS index JSON representation")

    chain_ids: list[str] = []
    atom_counts: dict[str, int] = {}
    for record in chain_records:
        chain_id = str(record.get("chain_id") or record.get("chain") or record.get("id") or "")
        if chain_id == "":
            raise ValueError("SAXS index record is missing chain identifier")
        atom_count = int(record.get("atom_count", record.get("count", 0)))
        chain_ids.append(chain_id)
        atom_counts[chain_id] = atom_count

    c1 = container.get("c1_used", container.get("c1"))
    c2 = container.get("c2_used", container.get("c2"))
    return {
        "chain_ids": np.asarray(chain_ids, dtype=str),
        "atom_counts": atom_counts,
        "c1_used": None if c1 is None else float(c1),
        "c2_used": None if c2 is None else float(c2),
    }


def load_hdx_index(path: str | Path) -> dict[str, np.ndarray | None]:
    """Read and normalize an HDX index JSON file."""
    return parse_hdx_index_payload(_load_json(Path(path)))


def load_saxs_index(path: str | Path) -> dict[str, Any]:
    """Read and normalize a SAXS index JSON file."""
    return parse_saxs_index_payload(_load_json(Path(path)))

