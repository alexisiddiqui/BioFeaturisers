import os
import pytest
from biofeaturisers.env import ComputeConfig, Device, Backend

@pytest.fixture(autouse=True)
def clean_jax_env():
    """Clear JAX-related environment variables before each test."""
    for key in ["JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE"]:
        if key in os.environ:
            del os.environ[key]
    yield

def test_compute_config_cpu():
    config = ComputeConfig(device=Device.CPU, disable_preallocation=True)
    config.configure()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"

def test_compute_config_gpu():
    config = ComputeConfig(device=Device.GPU, gpu_index=1)
    config.configure()
    assert os.environ.get("JAX_PLATFORMS") == "cuda"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
