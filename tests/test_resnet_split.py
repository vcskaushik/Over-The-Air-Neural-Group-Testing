"""Regression tests: refactored encode/channel/decode must compose to the original _forward_impl output."""
import pytest
import torch
import resnet_design2 as models


@pytest.fixture
def model_no_phase():
    torch.manual_seed(0)
    m = models.resnet18(pretrained=False, gt=True, phase=False)
    m.eval()
    return m


@pytest.fixture
def model_phase():
    torch.manual_seed(0)
    m = models.resnet18(pretrained=False, gt=True, phase=True)
    m.eval()
    return m


def _input(B=2, K=1):
    torch.manual_seed(42)
    return torch.randn(B, K, 3, 224, 224)


def test_resnet_gt_split_matches_monolithic(model_no_phase):
    """encode -> channel -> decode must equal the original _forward_impl output."""
    x = _input(B=2, K=1)
    with torch.no_grad():
        logits_full, mean_full, power_full = model_no_phase(x, noise_std=None, gpu=None)
        pre = model_no_phase.encode(x)
        post, mean_split, power_split = model_no_phase.channel(pre, noise_std=None, gpu=None)
        logits_split = model_no_phase.decode(post)
    assert torch.allclose(logits_full, logits_split, atol=1e-6)
    assert torch.allclose(mean_full, mean_split, atol=1e-6)
    assert torch.allclose(power_full, power_split, atol=1e-6)


def test_encode_output_shape(model_no_phase):
    """encode(x) returns (B, K, C', H', W') with H'=W'=28 for 224 input."""
    x = _input(B=2, K=3)
    with torch.no_grad():
        pre = model_no_phase.encode(x)
    assert pre.shape == (2, 3, 128, 28, 28)  # ResNet-18 layer2 has 128 channels


def test_channel_sums_K(model_no_phase):
    """channel sums over the K dim and divides by K."""
    pre = torch.ones(2, 4, 8, 4, 4)  # (B, K, C, H, W)
    with torch.no_grad():
        post, _, _ = model_no_phase.channel(pre, noise_std=None, gpu=None)
    assert post.shape == (2, 8, 4, 4)
    assert torch.allclose(post, torch.ones(2, 8, 4, 4))  # sum=4*1, /K=4 -> 1


def test_channel_adds_noise_when_snr_given(model_no_phase):
    """With noise_std > 0, post-channel feature differs from noiseless case."""
    torch.manual_seed(7)
    pre = torch.ones(2, 4, 8, 4, 4)
    with torch.no_grad():
        post_clean, _, _ = model_no_phase.channel(pre, noise_std=None, gpu=None)
        post_noisy, _, _ = model_no_phase.channel(pre, noise_std=0.5, gpu=None)
    assert not torch.allclose(post_clean, post_noisy, atol=1e-3)


def test_resnet_gt_phase_split_matches_monolithic(model_phase):
    """Same regression test for ResNet_GT_phase.

    Alternating_batch_phase_operation draws random phase shifts, so we must fix
    the RNG seed identically before both calls to get a reproducible comparison.
    """
    x = _input(B=2, K=1)
    with torch.no_grad():
        torch.manual_seed(99)
        logits_full, mean_full, power_full = model_phase(x, noise_std=None, gpu=None)
        pre = model_phase.encode(x)
        torch.manual_seed(99)
        post, mean_split, power_split = model_phase.channel(pre, noise_std=None, gpu=None)
        logits_split = model_phase.decode(post)
    assert torch.allclose(logits_full, logits_split, atol=1e-6)
    assert torch.allclose(mean_full, mean_split, atol=1e-6)
    assert torch.allclose(power_full, power_split, atol=1e-6)
