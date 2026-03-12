import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Backend(str, Enum):
    JAX   = "jax"
    TORCH = "torch"   # deferred — not yet implemented

class Device(str, Enum):
    CPU  = "cpu"
    GPU  = "gpu"      # uses first available CUDA/ROCm device
    AUTO = "auto"     # GPU if available, else CPU

@dataclass
class ComputeConfig:
    backend:           Backend = Backend.JAX

    # Device selection
    device:            Device  = Device.AUTO
    gpu_index:         int     = 0         # which GPU when multiple are present

    # JAX memory behaviour
    # False (default): JAX preallocates 75% of GPU VRAM on first use.
    # True: disables preallocation — JAX allocates as needed.
    # Recommended True on shared HPC nodes; False for throughput-critical jobs.
    disable_preallocation: bool = False

    # JAX precision
    enable_x64:        bool    = False     # float64 support (slower on consumer GPUs)

    def configure(self) -> None:
        """Apply settings.  Call once before any JAX computation."""
        if self.backend == Backend.JAX:
            self._configure_jax()

    def _configure_jax(self) -> None:
        # Must be set before jax is imported to take effect
        if self.device == Device.CPU:
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        elif self.device == Device.GPU:
            os.environ.setdefault("JAX_PLATFORMS", "cuda")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu_index))
        # AUTO: let JAX pick; CUDA if available, else CPU

        if self.disable_preallocation:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        
        if self.enable_x64:
            import jax
            jax.config.update("jax_enable_x64", True)
