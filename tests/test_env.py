"""Tests for compute runtime configuration."""

from __future__ import annotations

import os
import sys
import types

import pytest

from biofeaturisers.env import Backend, ComputeConfig, Device


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def update(self, key: str, value: object) -> None:
        self.calls.append((key, value))


@pytest.fixture
def fake_jax(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    fake_module = types.SimpleNamespace(config=recorder)
    monkeypatch.setitem(sys.modules, "jax", fake_module)
    return recorder


def test_compute_config_sets_cpu_env(monkeypatch: pytest.MonkeyPatch, fake_jax: _Recorder) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    cfg = ComputeConfig(device=Device.CPU)
    cfg.configure()
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert ("jax_enable_x64", False) in fake_jax.calls


def test_compute_config_sets_gpu_env(monkeypatch: pytest.MonkeyPatch, fake_jax: _Recorder) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = ComputeConfig(device=Device.GPU, gpu_index=2)
    cfg.configure()
    assert os.environ["JAX_PLATFORMS"] == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert ("jax_enable_x64", False) in fake_jax.calls


def test_compute_config_disables_preallocation(
    monkeypatch: pytest.MonkeyPatch, fake_jax: _Recorder
) -> None:
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    cfg = ComputeConfig(device=Device.AUTO, disable_preallocation=True, enable_x64=True)
    cfg.configure()
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert ("jax_enable_x64", True) in fake_jax.calls


def test_torch_backend_not_implemented() -> None:
    cfg = ComputeConfig(backend=Backend.TORCH)
    with pytest.raises(NotImplementedError):
        cfg.configure()


def test_gpu_index_validation() -> None:
    with pytest.raises(ValueError):
        ComputeConfig(gpu_index=-1)
