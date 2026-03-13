"""Tests for HDX/SAXS configuration dataclasses."""

from __future__ import annotations

import pytest

from biofeaturisers.config import HDXConfig, SAXSConfig


def test_hdx_config_defaults() -> None:
    cfg = HDXConfig()
    assert cfg.beta_c == 0.35
    assert cfg.beta_h == 2.0
    assert cfg.beta_0 == 0.0
    assert cfg.chunk_size == 0
    assert cfg.batch_size == 8
    assert cfg.exchange_mask == []


def test_hdx_config_default_lists_are_not_shared() -> None:
    first = HDXConfig()
    second = HDXConfig()
    first.timepoints.append(1.0)
    first.exchange_mask.append("A:42")
    assert second.timepoints == []
    assert second.exchange_mask == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cutoff_c": 0.0},
        {"cutoff_h": -1.0},
        {"steepness_c": 0.0},
        {"steepness_h": -1.0},
        {"seq_sep_min": -1},
        {"chunk_size": -1},
        {"batch_size": 0},
        {"hdxrate_temp": 0.0},
        {"timepoints": [-1.0]},
    ],
)
def test_hdx_config_validates_ranges(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        HDXConfig(**kwargs)


def test_saxs_config_defaults() -> None:
    cfg = SAXSConfig()
    assert cfg.q_min == 0.01
    assert cfg.q_max == 0.5
    assert cfg.n_q == 300
    assert cfg.ff_table == "waasmaier_kirfel"
    assert cfg.chunk_size == 512


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_min": 0.0},
        {"q_min": 0.5, "q_max": 0.5},
        {"n_q": 0},
        {"chunk_size": 0},
        {"batch_size": 0},
        {"c1_steps": 0},
        {"c2_steps": 0},
        {"c1_range": (1.0, 1.0)},
        {"c2_range": (2.0, 1.0)},
        {"ff_table": "unknown"},
    ],
)
def test_saxs_config_validates_ranges(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SAXSConfig(**kwargs)

