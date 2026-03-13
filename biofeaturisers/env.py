"""Runtime compute configuration helpers."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    """Supported compute backends."""

    JAX = "jax"
    TORCH = "torch"


class Device(str, Enum):
    """Supported runtime device targets."""

    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


@dataclass(slots=True)
class ComputeConfig:
    """Configure backend and runtime behavior before heavy numerical work."""

    backend: Backend = Backend.JAX
    device: Device = Device.AUTO
    gpu_index: int = 0
    disable_preallocation: bool = False
    enable_x64: bool = False

    def __post_init__(self) -> None:
        if self.gpu_index < 0:
            raise ValueError("gpu_index must be >= 0")

    def configure(self) -> None:
        """Apply runtime settings."""
        if self.backend is Backend.JAX:
            self._configure_jax()
            return
        if self.backend is Backend.TORCH:
            raise NotImplementedError("Torch backend is not implemented yet.")
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _configure_jax(self) -> None:
        if self.device is Device.CPU:
            os.environ["JAX_PLATFORMS"] = "cpu"
        elif self.device is Device.GPU:
            os.environ["JAX_PLATFORMS"] = "cuda"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)

        if self.disable_preallocation:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", self.enable_x64)

