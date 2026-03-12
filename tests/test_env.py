import os
from biofeaturisers.env import ComputeConfig

def test_compute_config_cpu():
    config = ComputeConfig(device="cpu", xla_preallocate=False)
    config.apply()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"

def test_compute_config_gpu():
    config = ComputeConfig(device="gpu", xla_memory_fraction=0.8)
    config.apply()
    assert os.environ.get("JAX_PLATFORMS") == "cuda,cpu"
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "0.8"
