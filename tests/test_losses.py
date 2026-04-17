"""Unit tests for privacy loss functions."""
import math
import pytest
import torch
import torch.nn.functional as F

from privacy.losses import priv_loss_ce, priv_loss_entropy, priv_loss_entropy_multilabel


def test_priv_loss_ce_negates_cross_entropy():
    """priv_loss_ce should equal -F.cross_entropy on the same inputs."""
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    target = torch.tensor([3, 7, 1, 9])
    expected = -F.cross_entropy(logits, target)
    actual = priv_loss_ce(logits, target)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_priv_loss_entropy_uniform_logits_returns_minus_log_K():
    """Uniform logits -> H(p) = log K -> priv_loss_entropy = -log K."""
    K = 1000
    logits = torch.zeros(2, K)  # uniform after softmax
    actual = priv_loss_entropy(logits)
    expected = torch.tensor(-math.log(K))
    assert torch.allclose(actual, expected, atol=1e-5)


def test_priv_loss_entropy_one_hot_returns_zero():
    """One-hot logits -> H(p) ~ 0 -> priv_loss_entropy ~ 0."""
    logits = torch.zeros(2, 5)
    logits[:, 0] = 100.0  # softmax effectively one-hot at index 0
    actual = priv_loss_entropy(logits)
    assert torch.allclose(actual, torch.tensor(0.0), atol=1e-5)


def test_priv_loss_entropy_is_bounded_below_by_minus_log_K():
    """For any logits, priv_loss_entropy >= -log K (equality at uniform)."""
    torch.manual_seed(1)
    K = 50
    logits = torch.randn(8, K)
    actual = priv_loss_entropy(logits).item()
    assert actual >= -math.log(K) - 1e-5


def test_priv_loss_entropy_multilabel_uniform_returns_minus_K_log2():
    """All sigmoid probs at 0.5 -> per-class H = log 2 -> sum = -K log 2 (negated, summed)."""
    K = 100
    logits = torch.zeros(2, K)  # sigmoid(0) = 0.5
    actual = priv_loss_entropy_multilabel(logits)
    expected = torch.tensor(-K * math.log(2))
    assert torch.allclose(actual, expected, atol=1e-4)


def test_priv_loss_entropy_multilabel_extreme_returns_zero():
    """Sigmoid probs at 0 or 1 -> per-class H = 0 -> total = 0."""
    logits = torch.full((2, 50), 100.0)
    actual = priv_loss_entropy_multilabel(logits)
    assert torch.allclose(actual, torch.tensor(0.0), atol=1e-5)


def test_priv_losses_have_grad():
    """All losses must produce a gradient with respect to logits."""
    torch.manual_seed(2)
    logits = torch.randn(4, 10, requires_grad=True)
    target = torch.tensor([0, 1, 2, 3])

    for loss_fn, args in (
        (priv_loss_ce, (logits, target)),
        (priv_loss_entropy, (logits,)),
        (priv_loss_entropy_multilabel, (logits,)),
    ):
        if logits.grad is not None:
            logits.grad.zero_()
        loss = loss_fn(*args)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0
