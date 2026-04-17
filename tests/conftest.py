"""Shared pytest fixtures and markers for the OTA-NGT test suite."""
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end tests that take > 1 minute or need real data",
    )
    config.addinivalue_line(
        "markers",
        "gpu: tests that require an available CUDA device",
    )


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def small_batch():
    """A (B=2, K=1, C=3, H=64, W=64) random tensor for shape tests on CPU."""
    return torch.randn(2, 1, 3, 64, 64)


@pytest.fixture
def tiny_postchannel_resnext():
    """A (B=2, 512, 28, 28) random tensor matching ResNeXt-101 layer2 output."""
    return torch.randn(2, 512, 28, 28)


@pytest.fixture
def tiny_postchannel_resnet18():
    """A (B=2, 128, 28, 28) random tensor matching ResNet-18 layer2 output."""
    return torch.randn(2, 128, 28, 28)
