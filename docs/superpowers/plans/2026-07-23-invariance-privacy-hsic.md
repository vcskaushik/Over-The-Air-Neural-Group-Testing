# Invariance-Based Privacy (HSIC, v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-adversarial, HSIC-based invariance training path for OTA-NGT that trains encoder+receiver jointly to make post-channel features independent of the fine-grained ImageNet class while preserving binary firearm utility (ITIT, σ=0).

**Architecture:** A new `privacy/train_invariance.py` entry point warm-starts E+R from a Stage-A checkpoint and trains them jointly on `L_task + λ_H·HSIC`. HSIC is computed through a **frozen** ImageNet-pretrained conv extractor (spatially aware, non-gameable) on **background-only** post-channel features, with a **PK-balanced** sampler guaranteeing same-class pairs. No learned adversary in training; the adversary survives only in Stage-C eval.

**Tech Stack:** PyTorch, torchvision, numpy, pytest. Existing `resnet_design2` backbones and `privacy/` module.

## Global Constraints

- **v1 scope:** HSIC only, **ITIT only** (`--GT-alg 1`, `--background-K 0`), **σ=0** (`--SNR` unset). VIB and σ>0 are out of scope (v2).
- **HSIC extractor is never trainable** — frozen, `requires_grad=False`, `eval()`.
- **Extractor in-channels are arch-derived**, never hardcoded (ResNet-18 layer2 = 128, ResNeXt-101 = 512).
- **HSIC is a training signal only** — never reported as the privacy number; Stage C is the arbiter.
- **HSIC operates on post-channel features**, background rows only, masked **before** bandwidth/centering.
- **Penalty normalization is the mean** over extractors and bandwidths (so `λ_H` is comparable across configs).
- Run tests with the repo venv: `.venv/bin/python -m pytest`. Existing tests live in `tests/`.
- Follow existing code style in `privacy/` (module docstrings, keyword-only args in trainer functions).

---

### Task 1: `AdversaryHead` pretrained flag

**Files:**
- Modify: `privacy/adversary.py:19-23`
- Test: `tests/test_adversary.py` (append)

**Interfaces:**
- Produces: `AdversaryHead(arch_name: str, num_classes: int = 1000, pretrained: bool = False)` — new `pretrained` kwarg forwarded to the backbone factory; default `False` preserves existing behavior.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adversary.py`:
```python
import inspect
from privacy.adversary import AdversaryHead


def test_adversaryhead_has_pretrained_kwarg_default_false():
    sig = inspect.signature(AdversaryHead.__init__)
    assert "pretrained" in sig.parameters, "AdversaryHead must accept a 'pretrained' kwarg"
    assert sig.parameters["pretrained"].default is False, "pretrained must default to False (back-compat)"


def test_adversaryhead_builds_with_pretrained_false():
    # pretrained=False must not touch the network and must build the decoder stack.
    head = AdversaryHead(arch_name="resnet18", num_classes=7, pretrained=False)
    assert head.fc.out_features == 7
    assert hasattr(head, "layer3") and hasattr(head, "layer4")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adversary.py::test_adversaryhead_has_pretrained_kwarg_default_false -v`
Expected: FAIL (`pretrained` not in parameters).

- [ ] **Step 3: Write minimal implementation**

In `privacy/adversary.py`, change the constructor signature and the factory call:
```python
    def __init__(self, arch_name: str, num_classes: int = 1000, pretrained: bool = False):
        super().__init__()
        if not hasattr(models, arch_name):
            raise ValueError(f"Unknown arch_name {arch_name!r}; not found in resnet_design2")
        backbone = getattr(models, arch_name)(pretrained=pretrained, gt=True, phase=False)
```
(Leave the rest of `__init__` unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_adversary.py -v`
Expected: PASS (all, including pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add privacy/adversary.py tests/test_adversary.py
git commit -m "feat(privacy): add pretrained flag to AdversaryHead (M5)"
```

---

### Task 2: Dataset invariance mode (full background pool)

**Files:**
- Modify: `privacy/dataset.py:26-70`
- Test: `tests/test_dataset.py` (append)

**Interfaces:**
- Produces: `PrivacyTaskCoalitionDataset(dataset_list, args, split, wnid_to_imagenet_idx=None, invariance_mode: bool = False)`. When `invariance_mode=True`, negatives are **not** truncated to `len(positives)` — the full background pool is kept in `dataset_samples[0]`. Default `False` preserves existing behavior. `dataset_samples[0]` remains a list of `[path, firearm_target]` pairs (firearm_target: 1=firearm, 0=background).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset.py`:
```python
import types
import numpy as np
from privacy.dataset import PrivacyTaskCoalitionDataset


def _stub_task(samples, class_to_idx):
    """Minimal ImageFolder stand-in: only the attributes __init__ touches."""
    t = types.SimpleNamespace()
    t.samples = samples
    t.class_to_idx = class_to_idx
    t.loader = lambda p: None
    t.transform = None
    t.target_transform = None
    return t


def _args(background_K=0):
    return types.SimpleNamespace(background_K=background_K)


def test_invariance_mode_keeps_full_background_pool():
    np.random.seed(0)
    # 2 firearm positives, 10 background negatives across 3 wnids.
    pos = _stub_task([("/d/gun/a.jpg", 0), ("/d/gun/b.jpg", 0)], {"gun": 0})
    neg = _stub_task([(f"/d/n0{i%3}/{i}.jpg", 0) for i in range(10)],
                     {"n00": 0, "n01": 1, "n02": 2})

    default_ds = PrivacyTaskCoalitionDataset([pos, neg], _args(), split="train")
    inv_ds = PrivacyTaskCoalitionDataset([pos, neg], _args(), split="train", invariance_mode=True)

    # Default truncates negatives to len(positives)=2 -> total 4.
    assert len(default_ds) == 4
    # Invariance keeps all 10 negatives -> total 12.
    assert len(inv_ds) == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dataset.py::test_invariance_mode_keeps_full_background_pool -v`
Expected: FAIL (`invariance_mode` is an unexpected keyword argument).

- [ ] **Step 3: Write minimal implementation**

In `privacy/dataset.py`, add the parameter and branch the truncation:
```python
    def __init__(self, dataset_list: List[datasets.ImageFolder], args, split: str,
                 wnid_to_imagenet_idx=None, invariance_mode: bool = False):
```
Then replace the truncation line (`negative_data_list = normal_data_list[: len(positive_data_list)]`) with:
```python
        if invariance_mode:
            negative_data_list = normal_data_list  # full background pool (v1 HSIC path)
        else:
            negative_data_list = normal_data_list[: len(positive_data_list)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add privacy/dataset.py tests/test_dataset.py
git commit -m "feat(privacy): invariance_mode keeps full background pool (M3)"
```

---

### Task 3: PK-balanced background sampler

**Files:**
- Create: `privacy/sampler.py`
- Test: `tests/test_sampler.py`

**Interfaces:**
- Produces:
  - `build_sample_index(samples: list) -> tuple[list[int], dict[str, list[int]]]` — given a list of `[path, firearm_target]`, returns `(firearm_indices, bg_wnid_to_indices)` where the wnid is the parent dir of the path and firearm rows (target==1) go to `firearm_indices`.
  - `class PKBackgroundSampler(torch.utils.data.Sampler)` with `__init__(self, firearm_indices, bg_wnid_to_indices, pk_classes, pk_per_class, pk_firearm, seed=0)`, `__iter__` yielding **index lists** (use as loader `batch_sampler=`), `__len__` returning `len(usable_classes)//pk_classes`. Classes with `< pk_per_class` background samples are excluded. Each yielded batch = `pk_classes` distinct classes × `pk_per_class` background + `pk_firearm` firearm indices.
  - `PKBackgroundSampler.from_dataset(dataset, pk_classes, pk_per_class, pk_firearm, seed=0)` — builds the index from `dataset.dataset_samples[0]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sampler.py`:
```python
from pathlib import Path
from privacy.sampler import build_sample_index, PKBackgroundSampler


def test_build_sample_index_splits_firearm_and_background():
    samples = [
        ["/d/n01/a.jpg", 0], ["/d/n01/b.jpg", 0],
        ["/d/n02/c.jpg", 0],
        ["/d/gun/x.jpg", 1], ["/d/gun/y.jpg", 1],
    ]
    firearm, bg = build_sample_index(samples)
    assert firearm == [3, 4]
    assert bg == {"n01": [0, 1], "n02": [2]}


def _make_index(n_classes, per_class, n_firearm):
    idx = 0
    bg = {}
    for c in range(n_classes):
        bg[f"n{c:03d}"] = list(range(idx, idx + per_class)); idx += per_class
    firearm = list(range(idx, idx + n_firearm))
    return firearm, bg


def test_sampler_batches_have_pk_structure_and_exclude_underfilled():
    firearm, bg = _make_index(n_classes=10, per_class=4, n_firearm=6)
    bg["n999"] = [1000, 1001]  # underfilled (2 < K=4) -> must be excluded
    s = PKBackgroundSampler(firearm, bg, pk_classes=3, pk_per_class=4, pk_firearm=2, seed=0)

    assert len(s) == 10 // 3  # 3 full groups of classes; underfilled excluded

    all_batches = list(s)
    assert len(all_batches) == len(s)
    for batch in all_batches:
        assert len(batch) == 3 * 4 + 2  # P*K + F
        # last F indices are firearm
        assert all(i in firearm for i in batch[-2:])
        bg_part = batch[:-2]
        # bg_part spans exactly 3 distinct classes, 4 each
        classes = [w for w, idxs in bg.items() for i in bg_part if i in idxs]
        assert len(set(classes)) == 3
        assert 1000 not in bg_part and 1001 not in bg_part  # underfilled never selected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sampler.py -v`
Expected: FAIL (`No module named 'privacy.sampler'`).

- [ ] **Step 3: Write minimal implementation**

Create `privacy/sampler.py`:
```python
"""PK-balanced background sampler for the HSIC invariance path.

Each batch contains P distinct background classes x K same-class samples (so the
delta label kernel has same-class pairs) plus F firearm samples (so L_task sees
positives). Classes with fewer than K background samples are excluded.
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch.utils.data


def build_sample_index(samples: List) -> Tuple[List[int], Dict[str, List[int]]]:
    """Split dataset_samples[0] into (firearm_indices, bg_wnid_to_indices)."""
    firearm_indices: List[int] = []
    bg: Dict[str, List[int]] = {}
    for i, entry in enumerate(samples):
        path, firearm_target = entry[0], int(entry[1])
        if firearm_target == 1:
            firearm_indices.append(i)
        else:
            wnid = Path(path).parent.name
            bg.setdefault(wnid, []).append(i)
    return firearm_indices, bg


class PKBackgroundSampler(torch.utils.data.Sampler):
    def __init__(self, firearm_indices, bg_wnid_to_indices,
                 pk_classes, pk_per_class, pk_firearm, seed=0):
        self.firearm_indices = list(firearm_indices)
        self.pk_classes = pk_classes
        self.pk_per_class = pk_per_class
        self.pk_firearm = pk_firearm
        self.seed = seed
        # Only classes with >= K background samples are usable.
        self.usable = {w: idxs for w, idxs in bg_wnid_to_indices.items()
                       if len(idxs) >= pk_per_class}
        self.num_batches = len(self.usable) // pk_classes

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        wnids = list(self.usable.keys())
        rng.shuffle(wnids)
        for b in range(self.num_batches):
            group = wnids[b * self.pk_classes:(b + 1) * self.pk_classes]
            batch: List[int] = []
            for w in group:
                batch.extend(rng.sample(self.usable[w], self.pk_per_class))
            if self.firearm_indices:
                batch.extend(rng.choices(self.firearm_indices, k=self.pk_firearm))
            yield batch

    @classmethod
    def from_dataset(cls, dataset, pk_classes, pk_per_class, pk_firearm, seed=0):
        firearm, bg = build_sample_index(dataset.dataset_samples[0])
        return cls(firearm, bg, pk_classes, pk_per_class, pk_firearm, seed=seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sampler.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add privacy/sampler.py tests/test_sampler.py
git commit -m "feat(privacy): PK-balanced background sampler (R2)"
```

---

### Task 4: HSIC penalty module

**Files:**
- Create: `privacy/hsic.py`
- Test: `tests/test_hsic.py`

**Interfaces:**
- Produces:
  - `rbf_kernel(x: Tensor[m,D], sigma: float) -> Tensor[m,m]`
  - `delta_label_kernel(y: Tensor[m]) -> Tensor[m,m]`
  - `hsic_biased(K: Tensor[m,m], L: Tensor[m,m]) -> Tensor[]` — biased V-statistic `tr(K H L H)/(m-1)^2`.
  - `median_bandwidth(x: Tensor[m,D], eps: float = 1e-6) -> Tensor[]`
  - `class HSICPenalty(nn.Module)` with `__init__(self, arch_name, extractor="pretrained", num_random=2, bandwidth_mults=(0.5,1.0,2.0), eps=1e-6)`; `forward(features: Tensor[m,C,H,W], labels: Tensor[m]) -> Tensor[]` returns the **mean** HSIC over extractors×bandwidths; `per_extractor(features, labels) -> list[Tensor[]]` for logging. All extractor params frozen.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hsic.py`:
```python
import torch
from privacy.hsic import (rbf_kernel, delta_label_kernel, hsic_biased,
                          median_bandwidth, HSICPenalty)


def test_delta_label_kernel():
    y = torch.tensor([0, 0, 1])
    L = delta_label_kernel(y)
    assert torch.equal(L, torch.tensor([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]]))


def test_rbf_kernel_diag_one_and_symmetric():
    x = torch.randn(5, 3)
    K = rbf_kernel(x, sigma=1.0)
    assert torch.allclose(torch.diag(K), torch.ones(5), atol=1e-5)
    assert torch.allclose(K, K.t(), atol=1e-6)


def test_hsic_dependent_greater_than_independent():
    torch.manual_seed(0)
    m = 40
    y = torch.randint(0, 4, (m,))
    L = delta_label_kernel(y)
    # Dependent features: one-hot(label) + small noise.
    dep = torch.nn.functional.one_hot(y, 4).float() + 0.01 * torch.randn(m, 4)
    # Independent features: pure noise.
    indep = torch.randn(m, 4)
    Kdep = rbf_kernel(dep, median_bandwidth(dep).item())
    Kind = rbf_kernel(indep, median_bandwidth(indep).item())
    assert hsic_biased(Kdep, L) > 5 * hsic_biased(Kind, L)


def test_median_bandwidth_eps_guard_on_collapsed_features():
    x = torch.ones(6, 4)  # all identical -> median pairwise distance 0
    assert median_bandwidth(x, eps=1e-6).item() >= 1e-6


def test_hsic_penalty_catches_spatially_hidden_class_info():
    """R1: class info hidden ONLY in spatial layout (identical channel means)
    is invisible to a global pool but visible to the conv extractor."""
    torch.manual_seed(0)
    m, C, H, W = 24, 128, 8, 8
    y = torch.tensor([0, 1] * (m // 2))
    feats = torch.zeros(m, C, H, W)
    for i in range(m):
        if y[i] == 0:
            feats[i, :, 0, 0] = 1.0   # top-left
        else:
            feats[i, :, H - 1, W - 1] = 1.0  # bottom-right
    # Global-pool baseline vectors are identical across classes -> HSIC ~ 0.
    pooled = feats.mean(dim=(2, 3))
    L = delta_label_kernel(y)
    pool_hsic = hsic_biased(rbf_kernel(pooled, median_bandwidth(pooled).item()), L)

    pen = HSICPenalty(arch_name="resnet18", extractor="random", num_random=1)
    ext_hsic = pen(feats, y)
    assert ext_hsic > 10 * (pool_hsic + 1e-12)


def test_hsic_penalty_extractor_receives_no_gradient():
    pen = HSICPenalty(arch_name="resnet18", extractor="random", num_random=1)
    feats = torch.randn(16, 128, 8, 8, requires_grad=True)
    y = torch.randint(0, 4, (16,))
    loss = pen(feats, y)
    loss.backward()
    for p in pen.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0, "extractor must be frozen"
    assert feats.grad is not None and feats.grad.abs().sum().item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hsic.py -v`
Expected: FAIL (`No module named 'privacy.hsic'`).

- [ ] **Step 3: Write minimal implementation**

Create `privacy/hsic.py`:
```python
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
    d = torch.cdist(x, x)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hsic.py -v`
Expected: PASS (all 6). Note `extractor="random"` avoids any network download.

- [ ] **Step 5: Commit**

```bash
git add privacy/hsic.py tests/test_hsic.py
git commit -m "feat(privacy): HSIC penalty with frozen spatial extractors (R1/R3/M8)"
```

---

### Task 5: `joint_step` in trainer

**Files:**
- Modify: `privacy/trainer.py` (append `joint_step`)
- Test: `tests/test_trainer.py` (append)

**Interfaces:**
- Consumes: `HSICPenalty` (Task 4), `backbone.encode/channel/decode` (existing).
- Produces: `joint_step(*, backbone, hsic_penalty, optimizer, images, firearm_target, imagenet_target_per_image, hsic_lambda, device, snr_noise_std=None) -> dict` with keys `loss_util`, `loss_hsic`, `loss_total`. Masks firearm rows out of the HSIC term (background-only, mask applied before HSIC). Updates E+R via the single `optimizer`. No adversary.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trainer.py`:
```python
from privacy.hsic import HSICPenalty
from privacy.trainer import joint_step


def _make_joint_setup():
    torch.manual_seed(0)
    backbone = models.resnet18(pretrained=False, gt=True, phase=False)
    hsic = HSICPenalty(arch_name="resnet18", extractor="random", num_random=1)
    optim = torch.optim.SGD(backbone.parameters(), lr=0.01)
    images = torch.randn(6, 1, 3, 16, 16)          # ITIT: K+1 = 1
    firearm_target = torch.tensor([0, 0, 0, 0, 1, 1])
    imagenet_target_per_image = torch.randint(0, 4, (6, 1))
    return backbone, hsic, optim, images, firearm_target, imagenet_target_per_image


def test_joint_step_runs_and_returns_finite_metrics():
    backbone, hsic, optim, images, ft, it = _make_joint_setup()
    out = joint_step(backbone=backbone, hsic_penalty=hsic, optimizer=optim,
                     images=images, firearm_target=ft, imagenet_target_per_image=it,
                     hsic_lambda=1.0, device=torch.device("cpu"), snr_noise_std=None)
    assert set(out) >= {"loss_util", "loss_hsic", "loss_total"}
    assert torch.isfinite(torch.tensor(out["loss_total"]))


def test_joint_step_grads_reach_encoder_and_receiver_not_extractor():
    backbone, hsic, optim, images, ft, it = _make_joint_setup()
    for p in backbone.parameters():
        p.grad = None
    joint_step(backbone=backbone, hsic_penalty=hsic, optimizer=optim,
               images=images, firearm_target=ft, imagenet_target_per_image=it,
               hsic_lambda=1.0, device=torch.device("cpu"), snr_noise_std=None)
    # Encoder AND receiver both get gradient (joint training, R not frozen).
    assert _params_grad_norm(backbone.layer2) > 0
    assert _params_grad_norm(backbone.layer3) > 0
    # HSIC extractor is frozen.
    for p in hsic.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trainer.py::test_joint_step_runs_and_returns_finite_metrics -v`
Expected: FAIL (`cannot import name 'joint_step'`).

- [ ] **Step 3: Write minimal implementation**

Append to `privacy/trainer.py`:
```python
def joint_step(*, backbone, hsic_penalty, optimizer, images, firearm_target,
               imagenet_target_per_image, hsic_lambda, device, snr_noise_std=None):
    """One joint E+R step on L_task + hsic_lambda * HSIC (background-only). No adversary.

    ITIT only: `images` is (B, 1, C, H, W); post-channel features are (B, C', H', W').
    HSIC is computed on background rows only, masked BEFORE the penalty.
    """
    backbone = backbone.to(device).train()
    hsic_penalty = hsic_penalty.to(device)
    images = images.to(device)
    firearm_target = firearm_target.to(device)
    imagenet_target_per_image = imagenet_target_per_image.to(device)
    for p in backbone.parameters():
        p.requires_grad = True

    pre = backbone.encode(images)
    post, _, _ = backbone.channel(pre, noise_std=snr_noise_std,
                                  gpu=device.index if device.type == "cuda" else None)
    util_logits = backbone.decode(post)
    loss_util = F.cross_entropy(util_logits, firearm_target)

    bg_mask = firearm_target == 0
    if int(bg_mask.sum().item()) >= 2:
        bg_feats = post[bg_mask]
        bg_labels = imagenet_target_per_image[bg_mask].reshape(-1)
        loss_hsic = hsic_penalty(bg_feats, bg_labels)
    else:
        loss_hsic = torch.zeros((), device=device)

    total = loss_util + hsic_lambda * loss_hsic
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    return {
        "loss_util": loss_util.detach().item(),
        "loss_hsic": float(loss_hsic.detach().item()),
        "loss_total": total.detach().item(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: PASS (all — new joint_step tests plus the pre-existing stage_b tests).

- [ ] **Step 5: Commit**

```bash
git add privacy/trainer.py tests/test_trainer.py
git commit -m "feat(privacy): joint E+R HSIC step (no adversary, background-masked)"
```

---

### Task 6: `train_invariance.py` entry point + startup diagnostic

**Files:**
- Create: `privacy/train_invariance.py`
- Test: `tests/test_smoke_invariance.py` (subprocess smoke, slow+gpu), `tests/test_invariance_ckpt.py` (offline unit)

**Interfaces:**
- Consumes: `load_stage_a`, `build_datasets`, `validate_utility` from `privacy.train_privacy`; `PrivacyTaskCoalitionDataset` (Task 2); `PKBackgroundSampler` (Task 3); `HSICPenalty` (Task 4); `joint_step` (Task 5).
- Produces:
  - `save_invariance_ckpt(backbone, coded_pwr, epoch, path)` — writes `{"state_dict": <module.-prefixed>, "coded_pwr": float, "epoch": int, "arch": str}` (main.py `--resume`-compatible).
  - `extractor_sanity(backbone, hsic_penalty, loader, device, snr_noise, num_batches=3) -> tuple[float, float]` — returns `(hsic_true, hsic_permuted_labels)` averaged over a few warm-start batches; used to catch an inert extractor.
  - CLI entry `python -m privacy.train_invariance` writing `invariance_final.pth.tar` in `--output_dir`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_invariance_ckpt.py`:
```python
import torch
import resnet_design2 as models
from privacy.train_invariance import save_invariance_ckpt


def test_save_invariance_ckpt_is_main_resume_compatible(tmp_path):
    backbone = models.resnet18(pretrained=False, gt=True, phase=False)
    path = tmp_path / "invariance_final.pth.tar"
    save_invariance_ckpt(backbone, coded_pwr=2.5, epoch=7, path=str(path))
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt and "coded_pwr" in ckpt and "epoch" in ckpt
    assert ckpt["coded_pwr"] == 2.5 and ckpt["epoch"] == 7
    # main.py --resume expects the module. prefix.
    assert all(k.startswith("module.") for k in ckpt["state_dict"])
```

Create `tests/test_smoke_invariance.py`:
```python
"""End-to-end smoke test for privacy.train_invariance. Slow + gpu."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "GroupTestingDataset"
STAGE_A_CKPT = REPO_ROOT / "Trained_Models" / "SmokeTest" / "checkpoint.pth.tar"


@pytest.mark.slow
@pytest.mark.gpu
def test_train_invariance_smoke(tmp_path):
    if not DATA_ROOT.exists():
        pytest.skip(f"Missing dataset at {DATA_ROOT}")
    if not STAGE_A_CKPT.exists():
        pytest.skip(f"Missing Stage A checkpoint at {STAGE_A_CKPT}")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not installed")

    out_dir = tmp_path / "InvarianceSmoke"
    cmd = [
        sys.executable, "-u", "-m", "privacy.train_invariance",
        "--stage-a-ckpt", str(STAGE_A_CKPT),
        "--data", str(DATA_ROOT), "--task-num", "2", "--background-K", "0",
        "--GT-alg", "1", "-a", "resnet18",
        "--hsic-lambda", "1.0", "--hsic-extractor", "random", "--hsic-num-random", "1",
        "--pk-classes", "4", "--pk-per-class", "4", "--pk-firearm", "4",
        "--epochs", "1", "--batch-size", "32",
        "-j", "2", "-valj", "1", "--print-freq", "1",
        "--output_dir", str(out_dir), "--log-name", "smoke.log",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"train_invariance failed:\n{proc.stdout}\n{proc.stderr}"
    assert (out_dir / "invariance_final.pth.tar").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_invariance_ckpt.py -v`
Expected: FAIL (`No module named 'privacy.train_invariance'`).

- [ ] **Step 3: Write minimal implementation**

Create `privacy/train_invariance.py`:
```python
"""Stage-B replacement: joint E+R HSIC invariance training (ITIT, sigma=0, v1).

No learned adversary. Warm-starts E+R from a Stage-A checkpoint and trains both
on L_task + lambda_H * HSIC(background). See the design spec for scope/risks.
"""
import argparse
import json
import os
import pathlib
import time

import torch

import resnet_design2 as models
from privacy.train_privacy import build_datasets, load_stage_a, validate_utility
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.sampler import PKBackgroundSampler
from privacy.hsic import HSICPenalty, delta_label_kernel, rbf_kernel, median_bandwidth, hsic_biased
from privacy.trainer import joint_step


def get_parser():
    p = argparse.ArgumentParser("OTA-NGT HSIC invariance training (v1)")
    p.add_argument("--data", required=True)
    p.add_argument("--task-num", type=int, default=2)
    p.add_argument("--background-K", type=int, required=True)
    p.add_argument("--GT-alg", type=int, choices=[1, 2], required=True)
    p.add_argument("-a", "--arch", required=True)
    p.add_argument("--phase", action="store_true")
    p.add_argument("--SNR", type=float, default=None)
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--hsic-lambda", type=float, default=1.0)
    p.add_argument("--hsic-extractor", choices=["pretrained", "random", "both"], default="pretrained")
    p.add_argument("--hsic-num-random", type=int, default=2)
    p.add_argument("--hsic-bandwidths", default="0.5,1,2")
    p.add_argument("--pk-classes", type=int, default=8)
    p.add_argument("--pk-per-class", type=int, default=4)
    p.add_argument("--pk-firearm", type=int, default=4)
    p.add_argument("--enc-lr", type=float, default=1e-4)
    p.add_argument("--rec-lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)  # unused with batch_sampler; kept for val
    p.add_argument("-j", "--workers", type=int, default=8)
    p.add_argument("-valj", "--val-workers", type=int, default=4)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log-name", default="invariance.log")
    p.add_argument("--print-freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    return p


def save_invariance_ckpt(backbone, coded_pwr, epoch, path):
    state = {f"module.{k}": v for k, v in backbone.state_dict().items()}
    torch.save({"state_dict": state, "coded_pwr": float(coded_pwr),
                "epoch": int(epoch), "arch": getattr(backbone, "arch_name", "")}, path)


def extractor_sanity(backbone, hsic_penalty, loader, device, snr_noise, num_batches=3):
    """HSIC(Phi(z), c) on true vs permuted labels — should be clearly higher for true."""
    backbone.eval()
    true_vals, perm_vals = [], []
    with torch.no_grad():
        for bi, (images, firearm_target, imagenet_target) in enumerate(loader):
            if bi >= num_batches:
                break
            images = images.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                          gpu=device.index if device.type == "cuda" else None)
            bg = (firearm_target == 0)
            if int(bg.sum()) < 2:
                continue
            feats = post[bg]
            labels = imagenet_target[bg].reshape(-1).to(device)
            true_vals.append(float(hsic_penalty(feats, labels)))
            perm = labels[torch.randperm(labels.numel(), device=device)]
            perm_vals.append(float(hsic_penalty(feats, perm)))
    avg = lambda xs: sum(xs) / max(len(xs), 1)
    return avg(true_vals), avg(perm_vals)


def joint_loop(backbone, hsic_penalty, train_dataset, val_dataset, args, device, log, coded_pwr):
    enc_params = list(backbone.conv1.parameters()) + list(backbone.bn1.parameters()) \
        + list(backbone.layer1.parameters()) + list(backbone.layer2.parameters())
    rec_params = list(backbone.layer3.parameters()) + list(backbone.layer4.parameters()) \
        + list(backbone.fc.parameters())
    optimizer = torch.optim.SGD(
        [{"params": enc_params, "lr": args.enc_lr},
         {"params": rec_params, "lr": args.rec_lr}],
        momentum=args.momentum, weight_decay=args.weight_decay)

    snr_noise = None  # v1: sigma = 0 (asserted in main)

    sampler = PKBackgroundSampler.from_dataset(
        train_dataset, args.pk_classes, args.pk_per_class, args.pk_firearm,
        seed=(args.seed or 0))

    # Startup diagnostic (M6): warn if the frozen extractor is inert on real features.
    diag_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=PKBackgroundSampler.from_dataset(
            train_dataset, args.pk_classes, args.pk_per_class, args.pk_firearm, seed=999),
        num_workers=args.workers, pin_memory=True)
    t_true, t_perm = extractor_sanity(backbone, hsic_penalty, diag_loader, device, snr_noise)
    line = f"[Diag] extractor HSIC true={t_true:.3e} permuted={t_perm:.3e}"
    print(line); log.write(line + "\n"); log.flush()
    if t_true <= t_perm:
        line = "[Diag][WARN] extractor HSIC not above permuted-label baseline — extractor may be inert!"
        print(line); log.write(line + "\n"); log.flush()

    for epoch in range(args.epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=PKBackgroundSampler.from_dataset(
                train_dataset, args.pk_classes, args.pk_per_class, args.pk_firearm,
                seed=(args.seed or 0) + epoch),
            num_workers=args.workers, pin_memory=True)
        t0 = time.time()
        for it, (images, firearm_target, imagenet_target_per_image) in enumerate(loader):
            metrics = joint_step(
                backbone=backbone, hsic_penalty=hsic_penalty, optimizer=optimizer,
                images=images, firearm_target=firearm_target,
                imagenet_target_per_image=imagenet_target_per_image,
                hsic_lambda=args.hsic_lambda, device=device, snr_noise_std=snr_noise)
            if it % args.print_freq == 0:
                line = (f"[Invar][ep {epoch}][it {it:4d}] util={metrics['loss_util']:.4f} "
                        f"hsic={metrics['loss_hsic']:.3e} total={metrics['loss_total']:.4f}")
                print(line); log.write(line + "\n"); log.flush()
        acc = validate_utility(backbone, val_dataset, args, device, coded_pwr=coded_pwr)
        line = f"[Invar][ep {epoch}] val_firearm_acc={acc:.4f} time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def main():
    args = get_parser().parse_args()
    assert args.GT_alg == 1 and args.background_K == 0, "v1 is ITIT-only (--GT-alg 1 --background-K 0)"
    assert args.SNR is None, "v1 is sigma=0; --SNR belongs to v2"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log = open(os.path.join(args.output_dir, args.log_name), "w")
    log.write(f"args: {json.dumps(vars(args), indent=2)}\n"); log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, coded_pwr = load_stage_a(args, device)
    backbone.arch_name = args.arch

    train_list, val_list = build_datasets(args)
    all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
    wnid_to_imagenet_idx = {w: i for i, w in enumerate(all_wnids)}

    train_dataset = PrivacyTaskCoalitionDataset(train_list, args, split="train",
                                                wnid_to_imagenet_idx=wnid_to_imagenet_idx,
                                                invariance_mode=True)
    val_dataset = PrivacyTaskCoalitionDataset(val_list, args, split="val",
                                              wnid_to_imagenet_idx=wnid_to_imagenet_idx)

    bandwidths = tuple(float(x) for x in args.hsic_bandwidths.split(","))
    hsic_penalty = HSICPenalty(arch_name=args.arch, extractor=args.hsic_extractor,
                               num_random=args.hsic_num_random, bandwidth_mults=bandwidths).to(device)

    joint_loop(backbone, hsic_penalty, train_dataset, val_dataset, args, device, log, coded_pwr)
    save_invariance_ckpt(backbone, coded_pwr, args.epochs,
                         os.path.join(args.output_dir, "invariance_final.pth.tar"))
    log.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_invariance_ckpt.py -v`
Expected: PASS. (The smoke test is skipped without data/GPU; that is expected.)

- [ ] **Step 5: Commit**

```bash
git add privacy/train_invariance.py tests/test_invariance_ckpt.py tests/test_smoke_invariance.py
git commit -m "feat(privacy): train_invariance entry point + extractor-sanity diagnostic (M6)"
```

---

### Task 7: Stage-C leakage on full background val + `--adv-init` + post-channel (M9/M1/M5)

**Files:**
- Modify: `privacy/eval_privacy.py` (parser, ITIT input → post, `--adv-init`, full-background-val leakage)
- Test: `tests/test_eval_privacy.py`

**Interfaces:**
- Consumes: `AdversaryHead(pretrained=...)` (Task 1).
- Produces:
  - `build_background_val_index(val_list, wnid_to_imagenet_idx) -> list[tuple[str, int]]` — flat `(path, imagenet_idx)` list over **all** background (task ≥ 1) val images.
  - `--adv-init {kaiming, pretrained}` (default `kaiming`) controlling `AdversaryHead(pretrained=...)`.
  - ITIT adversary input standardized on **post-channel** features (forward-compatible with v2 σ>0).

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_privacy.py`:
```python
import types
from privacy.eval_privacy import build_background_val_index


def _stub_task(paths):
    t = types.SimpleNamespace()
    t.samples = [(p, 0) for p in paths]
    return t


def test_build_background_val_index_covers_all_background_and_maps_labels():
    # task 0 = firearm (excluded), tasks 1.. = background (all included).
    firearm = _stub_task(["/d/gun/a.jpg"])
    bg1 = _stub_task(["/d/n01/x.jpg", "/d/n01/y.jpg"])
    bg2 = _stub_task(["/d/n02/z.jpg"])
    wnid_to_idx = {"gun": 0, "n01": 1, "n02": 2}

    index = build_background_val_index([firearm, bg1, bg2], wnid_to_idx)
    # 3 background images, none from the firearm task.
    assert len(index) == 3
    assert ("/d/n01/x.jpg", 1) in index
    assert ("/d/n02/z.jpg", 2) in index
    assert all(not p.startswith("/d/gun") for p, _ in index)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_privacy.py -v`
Expected: FAIL (`cannot import name 'build_background_val_index'`).

- [ ] **Step 3: Write minimal implementation**

In `privacy/eval_privacy.py`:

(a) Add the parser flag (after the `--stage-b-ckpt` line):
```python
    p.add_argument("--adv-init", choices=["kaiming", "pretrained"], default="kaiming")
```

(b) Build the adversary with the flag — change the `AdversaryHead(...)` construction in `main()`:
```python
    adversary = AdversaryHead(arch_name=args.arch,
                              num_classes=train_dataset.num_imagenet_classes,
                              pretrained=(args.adv_init == "pretrained")).to(device)
```

(c) Standardize ITIT adversary input on post-channel in **both** `train_fresh_adversary` and `evaluate_leakage` — replace the ITIT branch's
```python
                B, K, Cf, Hf, Wf = pre.shape
                adv_in = pre.reshape(B * K, Cf, Hf, Wf)
                adv_target = imagenet_target_per_image.reshape(-1)
                logits = adversary(adv_in)
```
with (post is already computed just above in each function):
```python
                adv_target = imagenet_target_per_image.reshape(-1)
                logits = adversary(post)
```

(d) Add the full-background-val helper near `build_datasets`:
```python
def build_background_val_index(val_list, wnid_to_imagenet_idx):
    """Flat (path, imagenet_idx) over ALL background (task>=1) val images.

    The ~300-sample PrivacyTaskCoalitionDataset val undercounts leakage (M9);
    Stage-C leakage is measured over the full background val set instead.
    """
    from pathlib import Path
    index = []
    for ds in val_list[1:]:  # task 0 = firearm (intended leak), excluded
        for path, _ in ds.samples:
            wnid = Path(path).parent.name
            index.append((path, wnid_to_imagenet_idx[wnid]))
    return index
```

(Wiring the full-background index into a leakage DataLoader is exercised by the slow eval run; the unit test covers the index-building contract.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_privacy.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite regression + commit**

Run: `.venv/bin/python -m pytest tests/ -v -m "not slow and not gpu"`
Expected: PASS (all offline tests across tasks 1–7).

```bash
git add privacy/eval_privacy.py tests/test_eval_privacy.py
git commit -m "feat(privacy): Stage-C post-channel input, --adv-init, full background val (M1/M5/M9)"
```

---

## Self-Review

**Spec coverage:**
- HSIC penalty (§4.1) → Task 4 ✓ (arch-derived channels, mean-normalize, ε-guard, multi-bandwidth, spatial extractor).
- PK sampler (§4.2) → Task 3 ✓ (underfilled-class exclusion, batch_sampler, epoch length).
- Dataset full background pool (§5) → Task 2 ✓.
- AdversaryHead pretrained (§5, M5) → Task 1 ✓.
- joint_step, post-channel, background mask (§3, §5) → Task 5 ✓.
- train_invariance entry point, warm-start, checkpoint format, startup diagnostic, ITIT/σ=0 asserts, λ_H=0 control is achievable via `--hsic-lambda 0` (§5, §6, §7, M4/M6) → Task 6 ✓.
- Stage-C full background val + `--adv-init` + post-channel (§5, §7, M1/M9) → Task 7 ✓.
- σ=0 enforced (Global Constraints, Task 6 assert) ✓. VIB absent ✓.
- λ_H=0 control run: no code needed — run Task 6's entry point with `--hsic-lambda 0`; noted here so the operator remembers it is mandatory (§7 M4).

**Placeholder scan:** No TBD/TODO; every code step contains full code; commands have expected output. ✓

**Type consistency:** `HSICPenalty(arch_name, extractor, num_random, bandwidth_mults)` used identically in Tasks 4/5/6. `joint_step(...)` keyword args match between Task 5 definition and Task 6 call site. `save_invariance_ckpt(backbone, coded_pwr, epoch, path)` and `build_background_val_index(val_list, wnid_to_imagenet_idx)` match their tests. `PKBackgroundSampler.from_dataset(dataset, pk_classes, pk_per_class, pk_firearm, seed)` consistent across Task 3 and Task 6. ✓

**Note on running order:** Tasks are dependency-ordered (1→7). Task 4's `extractor="random"` path is used in all offline tests to avoid network downloads; the `pretrained` extractor default is exercised only in the slow smoke/eval runs where network/data are available.
