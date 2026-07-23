"""HSIC independence penalty computed through frozen, spatially-aware extractors.

The penalty targets I(z; c) directly and is scramble-invariant. Extractors are
NEVER trainable (frozen), so the penalty is a genuine dependence measure rather
than a gameable adversary. See docs/superpowers/specs/2026-07-23-*-design.md.
"""
from typing import List

import torch
import torch.nn as nn

import resnet_design2 as models


def rbf_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sq = torch.cdist(x, x) ** 2
    return torch.exp(-sq / (2.0 * sigma * sigma))


def delta_label_kernel(y: torch.Tensor) -> torch.Tensor:
    y = y.reshape(-1, 1)
    return (y == y.t()).float()


def median_bandwidth(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Compute in float64: on collapsed/near-collapsed inputs the median distance
    # is ~0, and clamping a float32 zero to a small `eps` (e.g. 1e-6) rounds the
    # stored value down below the requested floor (float32 cannot represent
    # 1e-6 exactly), silently defeating the guard. float64 avoids that.
    d = torch.cdist(x.double(), x.double())
    med = torch.median(d)
    return torch.clamp(med, min=eps)


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    m = K.shape[0]
    H = torch.eye(m, device=K.device, dtype=K.dtype) - 1.0 / m
    return torch.trace(K @ H @ L @ H) / ((m - 1) ** 2)


def _layer2_out_channels(arch_name: str) -> int:
    """Arch-derived layer2 output width (ResNet-18=128, ResNeXt-101=512)."""
    bb = getattr(models, arch_name)(pretrained=False, gt=True, phase=False)
    return bb.layer3[0].conv1.in_channels


class _DecoderExtractor(nn.Module):
    """Frozen pretrained layer3+layer4+avgpool -> pooled vector."""
    def __init__(self, arch_name):
        super().__init__()
        bb = getattr(models, arch_name)(pretrained=True, gt=True, phase=False)
        self.layer3, self.layer4, self.avgpool = bb.layer3, bb.layer4, bb.avgpool

    def forward(self, x):
        x = self.layer3(x); x = self.layer4(x); x = self.avgpool(x)
        return torch.flatten(x, 1)


class _RandomConvExtractor(nn.Module):
    """Frozen random conv stack -> pooled vector (mixes space into channels)."""
    def __init__(self, in_ch, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(in_ch, 256, kernel_size=3, stride=2, padding=1)
        for p in self.conv.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.1
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.act(self.conv(x)); x = self.pool(x)
        return torch.flatten(x, 1)


class HSICPenalty(nn.Module):
    def __init__(self, arch_name, extractor="pretrained", num_random=2,
                 bandwidth_mults=(0.5, 1.0, 2.0), eps=1e-6):
        super().__init__()
        self.bandwidth_mults = tuple(bandwidth_mults)
        self.eps = eps
        in_ch = _layer2_out_channels(arch_name)
        exts: List[nn.Module] = []
        if extractor in ("pretrained", "both"):
            exts.append(_DecoderExtractor(arch_name))
        if extractor in ("random", "both"):
            for r in range(num_random):
                exts.append(_RandomConvExtractor(in_ch, seed=1234 + r))
        if not exts:
            raise ValueError(f"extractor must be pretrained|random|both, got {extractor!r}")
        self.extractors = nn.ModuleList(exts)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def per_extractor(self, features, labels):
        L = delta_label_kernel(labels)
        out = []
        for ext in self.extractors:
            vec = ext(features)
            h0 = median_bandwidth(vec, self.eps)
            vals = [hsic_biased(rbf_kernel(vec, (mult * h0).item()), L)
                    for mult in self.bandwidth_mults]
            out.append(torch.stack(vals).mean())
        return out

    def forward(self, features, labels):
        vals = self.per_extractor(features, labels)
        return torch.stack(vals).mean()

    def train(self, mode=True):  # keep extractors in eval regardless of parent .train()
        return super().train(False)
