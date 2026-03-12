import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ComputeConfig:
    device: str = "cpu"
    xla_preallocate: bool = False
    xla_memory_fraction: Optional[float] = None
    
    def apply(self):
        if not self.xla_preallocate:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if self.xla_memory_fraction is not None:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.xla_memory_fraction)
        
        # Setting device is usually done via jax.config, but setting CUDA_VISIBLE_DEVICES
        # or similar might be needed before JAX initialization.
        if self.device == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
        elif self.device == "gpu":
            os.environ["JAX_PLATFORMS"] = "cuda,cpu" # Fallback to cpu
