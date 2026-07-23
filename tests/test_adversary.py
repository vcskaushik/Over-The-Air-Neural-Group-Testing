"""Shape + structure tests for AdversaryHead."""
import pytest
import torch

from privacy.adversary import AdversaryHead


def test_adversary_head_resnet18_shape(tiny_postchannel_resnet18):
    """ResNet-18 backbone: post-channel (B, 128, 28, 28) -> (B, 1000) logits."""
    torch.manual_seed(0)
    head = AdversaryHead(arch_name="resnet18", num_classes=1000)
    head.eval()
    with torch.no_grad():
        logits = head(tiny_postchannel_resnet18)
    assert logits.shape == (2, 1000)


def test_adversary_head_resnext101_shape(tiny_postchannel_resnext):
    """ResNeXt-101 backbone: post-channel (B, 512, 28, 28) -> (B, 1000) logits."""
    torch.manual_seed(0)
    head = AdversaryHead(arch_name="resnext101_32x8d", num_classes=1000)
    head.eval()
    with torch.no_grad():
        logits = head(tiny_postchannel_resnext)
    assert logits.shape == (2, 1000)


def test_adversary_head_custom_num_classes():
    """num_classes flag controls fc out_features."""
    head = AdversaryHead(arch_name="resnet18", num_classes=42)
    assert head.fc.out_features == 42


def test_adversary_head_does_not_have_encoder_layers():
    """Encoder layers (conv1, bn1, layer1, layer2) must be absent — adversary only does decode."""
    head = AdversaryHead(arch_name="resnet18", num_classes=1000)
    for attr in ("conv1", "bn1", "layer1", "layer2"):
        assert not hasattr(head, attr), f"AdversaryHead leaks encoder attr {attr}"


def test_adversary_head_parameters_are_trainable():
    """All AdversaryHead parameters require grad by default."""
    head = AdversaryHead(arch_name="resnet18", num_classes=1000)
    assert all(p.requires_grad for p in head.parameters())


import inspect


def test_adversaryhead_has_pretrained_kwarg_default_false():
    sig = inspect.signature(AdversaryHead.__init__)
    assert "pretrained" in sig.parameters, "AdversaryHead must accept a 'pretrained' kwarg"
    assert sig.parameters["pretrained"].default is False, "pretrained must default to False (back-compat)"


def test_adversaryhead_builds_with_pretrained_false():
    # pretrained=False must not touch the network and must build the decoder stack.
    head = AdversaryHead(arch_name="resnet18", num_classes=7, pretrained=False)
    assert head.fc.out_features == 7
    assert hasattr(head, "layer3") and hasattr(head, "layer4")
