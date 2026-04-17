# Privacy-Preserving OTA-NGT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a privacy-preserving training mode for OTA-NGT in which the encoder produces post-channel features that the legitimate receiver can use for binary firearm detection but a worst-case adversary (same architecture, ground-truth ImageNet labels) cannot use to recover the input image's fine-grained ImageNet class.

**Architecture:** New `privacy/` package alongside existing `main.py`. Reuses the existing `ResNet_GT` after a backwards-compatible refactor that splits its monolithic `_forward_impl` into `encode()`, `channel()`, and `decode()` methods so all three downstream consumers (receiver, training-time adversary, post-hoc adversary) can attach to the post-channel feature without re-running the encoder. Training is two stages: utility pretrain (existing `main.py`) → frozen-receiver privacy fine-tune with TTUR adversary inner loop → short utility recovery. Honest leakage is measured by training a fresh adversary from scratch on the frozen encoder.

**Tech Stack:** PyTorch 2.6 (torchvision, torch.utils.data), pytest for tests, scikit-learn for evaluation metrics. Reuses existing `resnet_design2`, `constants.py`, and `data/GroupTestingDataset` pipeline.

**Spec reference:** `docs/superpowers/specs/2026-04-17-privacy-preserving-ota-ngt-design.md`.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `resnet_design2/my_resnet.py` | **Modify** | Split `ResNet_GT._forward_impl` and `ResNet_GT_phase._forward_impl` into `encode()`, `channel()`, `decode()`. `_forward_impl` becomes a 3-line composition. Backward-compatible. |
| `privacy/__init__.py` | Create | Package marker; exports `AdversaryHead`, `priv_loss_*`, dataset class. |
| `privacy/adversary.py` | Create | `AdversaryHead(nn.Module)` — borrows `layer3 + layer4 + avgpool` from a built backbone, replaces fc with `Linear(in, num_classes)`. |
| `privacy/losses.py` | Create | `priv_loss_ce(logits, target)`, `priv_loss_entropy(logits)`, `priv_loss_entropy_multilabel(logits)`. |
| `privacy/dataset.py` | Create | `PrivacyTaskCoalitionDataset` — like `TaskCoalitionDataset_SuperImposing` but `__getitem__` also returns the K-hot ImageNet label tensor. |
| `privacy/trainer.py` | Create | `stage_b_step(...)` — one (k_adv inner + 1 outer) update on a minibatch. Pure function-of-modules so it's unit-testable. |
| `privacy/train_privacy.py` | Create | Stage B + A_recovery entry point. Loads Stage A checkpoint, drives the per-epoch loop, saves Stage B checkpoint. |
| `privacy/eval_privacy.py` | Create | Stage C entry point. Loads Stage B checkpoint, trains fresh adversary from scratch, dumps leakage metrics JSON. |
| `tests/__init__.py` | Create | Test package marker. |
| `tests/conftest.py` | Create | Shared fixtures: `requires_cuda` skip-marker, tiny-batch random-tensor factories. |
| `tests/test_resnet_split.py` | Create | Regression test that the refactored `_forward_impl` matches original output bit-for-bit. |
| `tests/test_adversary.py` | Create | `AdversaryHead` shape and parameter-count checks across arches. |
| `tests/test_losses.py` | Create | Numerical correctness of each privacy loss vs hand-computed values. |
| `tests/test_dataset.py` | Create | `PrivacyTaskCoalitionDataset` returns correct ImageNet label shapes / values from a synthetic dir. |
| `tests/test_trainer.py` | Create | Stage-B step: gradient flow check (only `E` and `A` get gradients; `R` doesn't). |
| `tests/test_smoke_train.py` | Create | End-to-end smoke: 1 epoch of Stage B on `data/GroupTestingDataset` runs to completion, no NaN. (Marked `@pytest.mark.slow`; requires GPU.) |
| `tests/test_smoke_eval.py` | Create | End-to-end smoke: Stage C runs to completion on a 1-epoch checkpoint. (Marked slow + GPU.) |
| `README.md` | **Modify** | New "Privacy-Preserving Training" section with example commands for Stage A → B → C. |

---

## Task 1: Set up testing infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1.1: Install pytest in the venv**

```bash
.venv/bin/pip install pytest
```

Expected: installs `pytest` (and `pluggy`, `iniconfig`, etc.). Verify with `.venv/bin/pytest --version` — should print `pytest 8.x.x` or newer.

- [ ] **Step 1.2: Create empty test package marker**

Write `tests/__init__.py` with content:

```python
```

(Empty file. Just makes `tests/` a package.)

- [ ] **Step 1.3: Create conftest.py with shared fixtures**

Write `tests/conftest.py`:

```python
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
```

- [ ] **Step 1.4: Verify pytest discovers the empty suite**

Run: `.venv/bin/pytest tests/ -v`

Expected: `no tests ran in 0.0Xs`. No errors.

- [ ] **Step 1.5: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "tests: bootstrap pytest infra with shared fixtures and markers"
```

---

## Task 2: Refactor ResNet_GT into encode / channel / decode (regression test)

The existing `ResNet_GT._forward_impl` does the encoder pass, the channel (sum-of-K + AWGN), and the decoder pass in one monolithic function. We need to split it so privacy training can attach the adversary at the post-channel point without re-running the encoder. The split must be backward-compatible — existing `main.py` calls into `_forward_impl` and must get bit-identical output.

**Files:**
- Create: `tests/test_resnet_split.py`
- Modify: `resnet_design2/my_resnet.py:337-390` (`ResNet_GT._forward_impl`)
- Modify: `resnet_design2/my_resnet.py:470-536` (`ResNet_GT_phase._forward_impl`)

- [ ] **Step 2.1: Write the failing regression test**

Write `tests/test_resnet_split.py`:

```python
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
    """Same regression test for ResNet_GT_phase."""
    x = _input(B=2, K=1)
    with torch.no_grad():
        logits_full, mean_full, power_full = model_phase(x, noise_std=None, gpu=None)
        pre = model_phase.encode(x)
        post, mean_split, power_split = model_phase.channel(pre, noise_std=None, gpu=None)
        logits_split = model_phase.decode(post)
    assert torch.allclose(logits_full, logits_split, atol=1e-6)
```

- [ ] **Step 2.2: Run tests — confirm they fail with `AttributeError`**

Run: `.venv/bin/pytest tests/test_resnet_split.py -v`

Expected: 5 FAILED with `AttributeError: 'ResNet_GT' object has no attribute 'encode'` (or similar). The methods don't exist yet.

- [ ] **Step 2.3: Refactor `ResNet_GT` (lines 337-390)**

Open `resnet_design2/my_resnet.py`. Replace the current `_forward_impl` and add the three new methods. Keep the existing `forward` wrapper unchanged.

Replace the existing `ResNet_GT._forward_impl` (currently lines 337-387) with:

```python
    def encode(self, x):
        """Encoder pass: takes (B, K, C, H, W), returns pre-channel features (B, K, C', H', W')."""
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        _, C2, H2, W2 = x.shape
        x = x.view(B, K, C2, H2, W2)
        return x

    def channel(self, pre_channel, noise_std=None, gpu=None):
        """Channel: sums K, optionally adds AWGN, divides by K. Returns (post-channel, mean, power)."""
        B, K, C, H, W = pre_channel.shape
        mean_x = torch.mean(pre_channel.detach())
        power_x = torch.mean(pre_channel.detach() ** 2)
        x = torch.sum(pre_channel, dim=1, keepdim=False)
        if noise_std is not None:
            noise = torch.normal(mean=0.0, std=noise_std, size=x.size()).detach()
            if gpu is not None:
                noise = noise.cuda(gpu, non_blocking=True)
            x = (x + noise) / K
        else:
            x = x / K
        return x, mean_x, power_x

    def decode(self, post_channel):
        """Decoder pass: takes (B, C', H', W'), returns class logits."""
        x = self.layer3(post_channel)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _forward_impl(self, x, noise_std=None, gpu=None):
        pre = self.encode(x)
        post, mean_x, power_x = self.channel(pre, noise_std, gpu)
        logits = self.decode(post)
        return logits, mean_x, power_x
```

- [ ] **Step 2.4: Refactor `ResNet_GT_phase` (lines 470-536) analogously**

The phase variant has additional logic between the encoder and the channel sum: it applies a random phase shift via `Alternating_batch_phase_operation` (defined at line 45). Keep that logic; it lives inside the `channel` method (since the phase shift is part of the channel model).

Replace the existing `ResNet_GT_phase._forward_impl` (currently lines 470-533) with:

```python
    def encode(self, x):
        """Encoder pass: takes (B, K, C, H, W), returns pre-channel features (B, K, C', H', W')."""
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        _, C2, H2, W2 = x.shape
        x = x.view(B, K, C2, H2, W2)
        return x

    def channel(self, pre_channel, noise_std=None, gpu=None):
        """Channel with random phase shift applied per-batch before summation."""
        B, K, C, H, W = pre_channel.shape
        mean_x = torch.mean(pre_channel.detach())
        power_x = torch.mean(pre_channel.detach() ** 2)
        # Apply random phase to the K dim, then sum (existing _phase behavior)
        x = Alternating_batch_phase_operation(pre_channel)
        x = torch.sum(x, dim=1, keepdim=False)
        if noise_std is not None:
            noise = torch.normal(mean=0.0, std=noise_std, size=x.size()).detach()
            if gpu is not None:
                noise = noise.cuda(gpu, non_blocking=True)
            x = (x + noise) / K
        else:
            x = x / K
        # Phase variant stores complex mag; take real part for decoder input.
        if torch.is_complex(x):
            x = x.real
        return x, mean_x, power_x

    def decode(self, post_channel):
        x = self.layer3(post_channel)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _forward_impl(self, x, noise_std=None, gpu=None):
        pre = self.encode(x)
        post, mean_x, power_x = self.channel(pre, noise_std, gpu)
        logits = self.decode(post)
        return logits, mean_x, power_x
```

> **Note:** Read the original `ResNet_GT_phase._forward_impl` body before pasting — if it does anything beyond the `Alternating_batch_phase_operation` + sum + noise that this `channel()` captures (e.g., reshapes, type casts), preserve those operations inside `channel()`. The regression test in step 2.6 will catch any mismatch.

- [ ] **Step 2.5: Run the regression tests — confirm they pass**

Run: `.venv/bin/pytest tests/test_resnet_split.py -v`

Expected: 5 PASSED. If `test_resnet_gt_phase_split_matches_monolithic` fails, inspect the diff carefully — the phase-variant probably has additional ops the channel() method needs to preserve. Patch and re-run.

- [ ] **Step 2.6: Verify main.py still trains (smoke check, no commit)**

Run a 5-iteration smoke check that the existing pipeline produces the same shape/loss numbers it did before:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u main.py \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 \
    --GT-alg 1 -a resnet18 --pretrained --epochs 1 --batch-size 16 \
    -j 4 -valj 2 --print-freq 1 --output_dir Trained_Models/RefactorCheck \
    --log-name refactor.log 2>&1 | head -30
```

Expected: see "No. 0, traindir ..." then a few `Epoch: [0][N/487]` lines with loss/acc consistent with the smoke test from earlier (loss starting around 0.6-0.7, acc around 50-80%). Kill it once you see ~5 batches print successfully (Ctrl-C if interactive, or just let `head -30` close the pipe — wget-style "head" closes the read end and `python -u` will SIGPIPE shortly after). No NaN, no traceback.

- [ ] **Step 2.7: Commit**

```bash
git add resnet_design2/my_resnet.py tests/test_resnet_split.py
git commit -m "refactor(resnet_gt): split _forward_impl into encode/channel/decode

Backwards-compatible refactor: _forward_impl is now a 3-line
composition of encode + channel + decode. Privacy training (next
commits) attaches the adversary at the post-channel tap point
without re-running the encoder. Regression test confirms bit-for-bit
parity with the previous monolithic implementation."
```

---

## Task 3: AdversaryHead module

**Files:**
- Create: `privacy/__init__.py`
- Create: `privacy/adversary.py`
- Create: `tests/test_adversary.py`

- [ ] **Step 3.1: Create empty package marker**

Write `privacy/__init__.py`:

```python
"""Privacy-preserving OTA-NGT training: adversary, losses, dataset, trainer."""
from .adversary import AdversaryHead

__all__ = ["AdversaryHead"]
```

- [ ] **Step 3.2: Write the failing test**

Write `tests/test_adversary.py`:

```python
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
```

- [ ] **Step 3.3: Run tests — confirm `ModuleNotFoundError`**

Run: `.venv/bin/pytest tests/test_adversary.py -v`

Expected: 5 ERRORS or FAILED with `ModuleNotFoundError: No module named 'privacy.adversary'` (or `ImportError` on `AdversaryHead`).

- [ ] **Step 3.4: Implement `AdversaryHead`**

Write `privacy/adversary.py`:

```python
"""AdversaryHead: same architecture as the OTA-NGT receiver (layer3 + layer4 + avgpool + fc),
but with a configurable-width final classifier head. Used for both the in-loop privacy adversary
during Stage B and the from-scratch honest-evaluation adversary in Stage C.
"""
import torch
import torch.nn as nn

import resnet_design2 as models


class AdversaryHead(nn.Module):
    """Decoder-only twin of `ResNet_GT`. Consumes post-channel features.

    Args:
        arch_name: name of a model factory in `resnet_design2` (e.g. "resnet18", "resnext101_32x8d").
        num_classes: output dimension of the final fc head.
    """

    def __init__(self, arch_name: str, num_classes: int = 1000):
        super().__init__()
        if not hasattr(models, arch_name):
            raise ValueError(f"Unknown arch_name {arch_name!r}; not found in resnet_design2")
        backbone = getattr(models, arch_name)(pretrained=False, gt=True, phase=False)
        # Borrow only the decoder slice. Drop encoder layers.
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        in_features = backbone.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)
        del backbone

    def forward(self, post_channel: torch.Tensor) -> torch.Tensor:
        x = self.layer3(post_channel)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
```

- [ ] **Step 3.5: Run tests — confirm 5 PASSED**

Run: `.venv/bin/pytest tests/test_adversary.py -v`

Expected: 5 PASSED.

- [ ] **Step 3.6: Commit**

```bash
git add privacy/__init__.py privacy/adversary.py tests/test_adversary.py
git commit -m "feat(privacy): add AdversaryHead — decoder-only twin of ResNet_GT receiver

Borrows layer3 + layer4 + avgpool from a built backbone, swaps fc for
configurable-width head. Used for both Stage B in-loop adversary and
Stage C honest-evaluation adversary."
```

---

## Task 4: Privacy loss functions

**Files:**
- Create: `privacy/losses.py`
- Create: `tests/test_losses.py`
- Modify: `privacy/__init__.py` (add exports)

- [ ] **Step 4.1: Write the failing tests**

Write `tests/test_losses.py`:

```python
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
```

- [ ] **Step 4.2: Run tests — confirm `ImportError`**

Run: `.venv/bin/pytest tests/test_losses.py -v`

Expected: ImportError on `priv_loss_ce` etc.

- [ ] **Step 4.3: Implement the losses**

Write `privacy/losses.py`:

```python
"""Privacy losses for the encoder's outer step in Stage B.

Two scientifically distinct forms (selected by config in Stage B):

* `priv_loss_ce`: negated adversary cross-entropy. Encoder is rewarded for making
  the adversary's prediction wrong (any way, including confidently wrong). Has a
  label-shift degenerate optimum but is the standard DANN-style baseline.

* `priv_loss_entropy` (single-label) and `priv_loss_entropy_multilabel`:
  negative entropy of the adversary's (per-class) softmax/sigmoid output. Encoder
  is rewarded for making the adversary's posterior maximally uncertain. No
  degenerate optimum; equivalent up to a constant to KL-to-uniform.
"""
import torch
import torch.nn.functional as F


def priv_loss_ce(adv_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negated adversary cross-entropy.

    The encoder MINIMIZES this loss, which equals MAXIMIZING the adversary's CE.
    """
    return -F.cross_entropy(adv_logits, target)


def priv_loss_entropy(adv_logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of adversary's softmax distribution (single-label, ITIT)."""
    log_p = F.log_softmax(adv_logits, dim=-1)
    p = log_p.exp()
    # H(p) = -sum p log p ; loss = -H = sum p log p, then mean over batch
    return (p * log_p).sum(dim=-1).mean()


def priv_loss_entropy_multilabel(adv_logits: torch.Tensor) -> torch.Tensor:
    """Negative sum of per-class Bernoulli entropies (multilabel, GTGT-FM).

    For each of K class outputs, computes H(Bernoulli(sigmoid(z_k))) and sums them.
    Loss = -H_total. Minimum value = -K * log 2 (when all probs = 0.5).
    """
    log_p = F.logsigmoid(adv_logits)            # log p_k
    log_one_minus_p = F.logsigmoid(-adv_logits)  # log(1 - p_k)
    p = log_p.exp()
    one_minus_p = log_one_minus_p.exp()
    # H_k = -[p log p + (1-p) log(1-p)] ; sum over classes ; loss = -sum_k H_k = sum_k [p log p + (1-p) log(1-p)]
    per_class = p * log_p + one_minus_p * log_one_minus_p
    return per_class.sum(dim=-1).mean()
```

- [ ] **Step 4.4: Run tests — confirm 7 PASSED**

Run: `.venv/bin/pytest tests/test_losses.py -v`

Expected: 7 PASSED.

- [ ] **Step 4.5: Update `privacy/__init__.py` to export losses**

Edit `privacy/__init__.py` to:

```python
"""Privacy-preserving OTA-NGT training: adversary, losses, dataset, trainer."""
from .adversary import AdversaryHead
from .losses import priv_loss_ce, priv_loss_entropy, priv_loss_entropy_multilabel

__all__ = [
    "AdversaryHead",
    "priv_loss_ce",
    "priv_loss_entropy",
    "priv_loss_entropy_multilabel",
]
```

- [ ] **Step 4.6: Commit**

```bash
git add privacy/losses.py privacy/__init__.py tests/test_losses.py
git commit -m "feat(privacy): add ce / entropy / multilabel-entropy privacy losses

Three scientifically distinct privacy terms for the encoder outer step:
- priv_loss_ce        : negated adversary CE (DANN-style; degenerate optimum allowed)
- priv_loss_entropy   : -H(softmax) for single-label ITIT adversary
- priv_loss_entropy_multilabel : -sum_k H(Bernoulli) for multilabel GTGT-FM adversary

Spec ablation: ITIT runs sweep ce vs entropy."
```

---

## Task 5: PrivacyTaskCoalitionDataset

The existing `TaskCoalitionDataset_SuperImposing` (in `main.py`) returns `(images, binary_target)` where the original ImageNet wnid is discarded during binary remap. Stage B and Stage C need the original ImageNet labels for the adversary. We add a sibling dataset class that preserves them.

**Files:**
- Create: `privacy/dataset.py`
- Create: `tests/test_dataset.py`
- Modify: `privacy/__init__.py` (add export)

- [ ] **Step 5.1: Write the failing tests**

Write `tests/test_dataset.py`:

```python
"""Tests for PrivacyTaskCoalitionDataset — must surface ImageNet labels alongside binary targets."""
import os
import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from privacy.dataset import PrivacyTaskCoalitionDataset


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Build a tiny GroupTestingDataset-like tree with 2 firearm classes and 4 background classes,
    each with 3 train images and 1 val image. Returns the root path."""
    root = tmp_path / "GroupTestingDataset"
    wnid_layout = {
        "0": ["n02749479", "n04086273"],          # 2 firearm classes
        "1": ["n01440764", "n01443537", "n01484850", "n01491361"],  # 4 background classes
    }
    counts = {"train": 3, "val": 1}
    for task, wnids in wnid_layout.items():
        for wnid in wnids:
            for split, n in counts.items():
                d = root / task / split / wnid
                d.mkdir(parents=True)
                for i in range(n):
                    img = Image.new("RGB", (16, 16), color=(i * 30, i * 30, i * 30))
                    img.save(d / f"{wnid}_{i:05d}.JPEG")
    return root


def _build_args(data_root, background_K=0):
    return Namespace(data=str(data_root), task_num=2, background_K=background_K)


def _build_dataset_list(data_root):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    out = []
    for task in ("0", "1"):
        out.append(datasets.ImageFolder(str(Path(data_root) / task / "train"), transform=transform))
    return out


def test_dataset_yields_three_outputs(synthetic_dataset):
    """__getitem__ must return (images, firearm_target, imagenet_targets) — three items."""
    args = _build_args(synthetic_dataset, background_K=0)
    dl = _build_dataset_list(synthetic_dataset)
    ds = PrivacyTaskCoalitionDataset(dl, args, split="train")
    item = ds[0]
    assert isinstance(item, tuple) and len(item) == 3


def test_imagenet_targets_shape_matches_K_plus_1(synthetic_dataset):
    """imagenet_targets is a tensor of shape (K+1,) with one ImageNet class per stacked image."""
    args = _build_args(synthetic_dataset, background_K=2)  # K=3
    dl = _build_dataset_list(synthetic_dataset)
    ds = PrivacyTaskCoalitionDataset(dl, args, split="train")
    images, firearm_target, imagenet_targets = ds[0]
    assert images.shape == (3, 3, 32, 32)
    assert imagenet_targets.shape == (3,)
    assert imagenet_targets.dtype == torch.long


def test_imagenet_label_is_consistent_with_path(synthetic_dataset):
    """The first image's imagenet label must match the wnid extracted from its path."""
    args = _build_args(synthetic_dataset, background_K=0)
    dl = _build_dataset_list(synthetic_dataset)
    ds = PrivacyTaskCoalitionDataset(dl, args, split="train")
    # All wnids the dataset knows about, sorted alphabetically (the canonical mapping).
    expected_wnids = sorted(["n02749479", "n04086273", "n01440764", "n01443537", "n01484850", "n01491361"])
    wnid_to_idx = {w: i for i, w in enumerate(expected_wnids)}

    for idx in range(len(ds)):
        path = ds.dataset_samples[0][idx][0]
        wnid = Path(path).parent.name
        _, _, imagenet_targets = ds[idx]
        assert imagenet_targets[0].item() == wnid_to_idx[wnid]


def test_firearm_target_remains_binary(synthetic_dataset):
    """firearm_target is still 0 or 1 (preserves parent dataset semantics)."""
    args = _build_args(synthetic_dataset, background_K=0)
    dl = _build_dataset_list(synthetic_dataset)
    ds = PrivacyTaskCoalitionDataset(dl, args, split="train")
    targets = {ds[i][1] for i in range(len(ds))}
    assert targets <= {0, 1}
```

- [ ] **Step 5.2: Run tests — confirm `ImportError`**

Run: `.venv/bin/pytest tests/test_dataset.py -v`

Expected: ImportError on `PrivacyTaskCoalitionDataset`.

- [ ] **Step 5.3: Implement `PrivacyTaskCoalitionDataset`**

Write `privacy/dataset.py`:

```python
"""PrivacyTaskCoalitionDataset: like main.py's TaskCoalitionDataset_SuperImposing,
but each __getitem__ also returns a (K+1,) tensor of ImageNet class indices for the
stacked images. The Stage B / Stage C adversary trains against these labels.
"""
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets


class PrivacyTaskCoalitionDataset(torch.utils.data.Dataset):
    """Wraps a list of `ImageFolder` datasets (one per task) and yields:

    * `images`            : (K+1, C, H, W) stacked transforms
    * `firearm_target`    : int in {0, 1} — binary firearm label
    * `imagenet_targets`  : long tensor of shape (K+1,) — per-image ImageNet class index

    The ImageNet class index is assigned as the alphabetical position of the wnid
    among all wnids present across all task folders. This makes the index stable
    across runs as long as the on-disk class folders don't change.
    """

    def __init__(self, dataset_list: List[datasets.ImageFolder], args, split: str):
        assert split in ("train", "val")

        first = dataset_list[0]
        self.loader = first.loader
        self.transform = first.transform
        assert first.target_transform is None, "PrivacyTaskCoalitionDataset assumes target_transform is None"
        self.args = args

        # Build the global wnid -> imagenet index mapping.
        all_wnids = sorted({wnid for ds in dataset_list for wnid in ds.class_to_idx.keys()})
        self.wnid_to_imagenet_idx = {w: i for i, w in enumerate(all_wnids)}
        self.num_imagenet_classes = len(all_wnids)

        # Same mixing logic as TaskCoalitionDataset_SuperImposing: positives = task 0, negatives = sampled background.
        positive_data_list = dataset_list[0].samples  # firearm
        normal_data_list = []
        for ds in dataset_list[1:]:
            normal_data_list.extend(ds.samples)
        normal_data_list = np.random.permutation(normal_data_list)

        negative_data_list = normal_data_list[: len(positive_data_list)]

        positive_target = 1
        positive_data_list = [[s[0], positive_target] for s in positive_data_list]
        negative_target = 0
        negative_data_list = [[s[0], negative_target] for s in negative_data_list]

        mixing_data_list = positive_data_list + negative_data_list
        if split != "val":
            mixing_data_list = list(np.random.permutation(mixing_data_list))

        self.background_K = self.args.background_K

        background_K_list = []
        for _ in range(self.background_K):
            sample = list(np.random.permutation(normal_data_list)[: len(mixing_data_list)])
            background_K_list.append(sample)

        self.dataset_samples = [mixing_data_list] + background_K_list

    def __len__(self) -> int:
        return len(self.dataset_samples[0])

    def __getitem__(self, index):
        firearm_target = int(self.dataset_samples[0][index][1])

        images = []
        imagenet_targets = []
        for folder_idx in range(self.background_K + 1):
            path, _ = self.dataset_samples[folder_idx][index]
            sample = self.loader(path)
            sample = self.transform(sample)
            images.append(sample)
            wnid = Path(path).parent.name
            imagenet_targets.append(self.wnid_to_imagenet_idx[wnid])

        images_stack = torch.stack(images)
        imagenet_targets_t = torch.tensor(imagenet_targets, dtype=torch.long)
        return images_stack, firearm_target, imagenet_targets_t
```

- [ ] **Step 5.4: Run tests — confirm 4 PASSED**

Run: `.venv/bin/pytest tests/test_dataset.py -v`

Expected: 4 PASSED.

- [ ] **Step 5.5: Update `privacy/__init__.py` to export the dataset**

Edit `privacy/__init__.py` to:

```python
"""Privacy-preserving OTA-NGT training: adversary, losses, dataset, trainer."""
from .adversary import AdversaryHead
from .dataset import PrivacyTaskCoalitionDataset
from .losses import priv_loss_ce, priv_loss_entropy, priv_loss_entropy_multilabel

__all__ = [
    "AdversaryHead",
    "PrivacyTaskCoalitionDataset",
    "priv_loss_ce",
    "priv_loss_entropy",
    "priv_loss_entropy_multilabel",
]
```

- [ ] **Step 5.6: Commit**

```bash
git add privacy/dataset.py privacy/__init__.py tests/test_dataset.py
git commit -m "feat(privacy): add PrivacyTaskCoalitionDataset preserving ImageNet labels

Stage B/C adversary needs the original ImageNet wnid for each stacked
image; main.py's TaskCoalitionDataset throws this away after binary
remap. The new class preserves them via a sorted-wnid global index and
returns (images, firearm_target, imagenet_targets) per __getitem__."
```

---

## Task 6: Stage-B training step (gradient flow contract)

This is the heart of the privacy training: a single (k_adv inner + 1 outer) update on a minibatch. We isolate it in `privacy/trainer.py:stage_b_step` so we can unit-test it without spinning up the full training loop.

**Files:**
- Create: `privacy/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 6.1: Write the failing test**

Write `tests/test_trainer.py`:

```python
"""Tests for stage_b_step: gradient flow contract and basic shape correctness."""
import pytest
import torch
import torch.nn as nn

import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.trainer import stage_b_step


def _params_grad_norm(module):
    """Total L2 norm of grads across all parameters (0 if all are None or zero)."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def _set_module_grads_to_none(module):
    for p in module.parameters():
        p.grad = None


@pytest.fixture
def tiny_setup():
    """Build a tiny end-to-end setup: ResNet-18 GT receiver, AdversaryHead, optimizers, fake batch.

    Uses CPU and 16x16 input so the test runs fast.
    """
    torch.manual_seed(0)
    backbone = models.resnet18(pretrained=False, gt=True, phase=False)
    adversary = AdversaryHead(arch_name="resnet18", num_classes=10)

    enc_params = list(backbone.conv1.parameters()) + list(backbone.bn1.parameters()) \
        + list(backbone.layer1.parameters()) + list(backbone.layer2.parameters())
    dec_params = list(backbone.layer3.parameters()) + list(backbone.layer4.parameters()) \
        + list(backbone.fc.parameters())

    enc_optim = torch.optim.SGD(enc_params, lr=0.01)
    adv_optim = torch.optim.SGD(adversary.parameters(), lr=0.01)

    # Fake minibatch: 2 stacked images per sample, 3 channels, 16x16. K+1 = 2.
    images = torch.randn(2, 2, 3, 16, 16)
    firearm_target = torch.tensor([0, 1])
    imagenet_target_per_image = torch.randint(0, 10, (2, 2))  # (B, K+1)

    return {
        "backbone": backbone,
        "adversary": adversary,
        "enc_optim": enc_optim,
        "adv_optim": adv_optim,
        "enc_params": enc_params,
        "dec_params": dec_params,
        "images": images,
        "firearm_target": firearm_target,
        "imagenet_target_per_image": imagenet_target_per_image,
    }


def test_stage_b_step_runs(tiny_setup):
    """stage_b_step completes without error and returns a metrics dict."""
    out = stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=2,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
    )
    assert "loss_util" in out and "loss_priv" in out and "loss_adv" in out
    assert torch.isfinite(torch.tensor(out["loss_util"]))


def test_stage_b_step_freezes_receiver(tiny_setup):
    """Receiver (layer3, layer4, fc) must get NO gradient during the step (frozen by design)."""
    backbone = tiny_setup["backbone"]
    # Pre-step: zero all grads
    _set_module_grads_to_none(backbone)
    _set_module_grads_to_none(tiny_setup["adversary"])

    stage_b_step(
        backbone=backbone,
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=1,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
    )

    # After step: encoder params (conv1, bn1, layer1, layer2) should have nonzero grads.
    for m in (backbone.conv1, backbone.layer1, backbone.layer2):
        assert _params_grad_norm(m) > 0, f"encoder layer {m.__class__.__name__} got no gradient"

    # Decoder (layer3, layer4, fc) must be untouched by the step (zeroed afterwards).
    for m in (backbone.layer3, backbone.layer4, backbone.fc):
        for p in m.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0, \
                f"decoder {m.__class__.__name__} got gradient — frozen-receiver contract broken"


def test_stage_b_step_priv_loss_ce_branch_runs(tiny_setup):
    """priv_loss_name='ce' takes the CE branch instead of entropy."""
    out = stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="ce",
        lam=0.5,
        k_adv=1,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
    )
    assert torch.isfinite(torch.tensor(out["loss_util"]))


def test_stage_b_step_gtgt_fm_branch_runs(tiny_setup):
    """gt_alg=2 takes the GTGT-FM (multilabel) branch."""
    out = stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=1,
        device=torch.device("cpu"),
        gt_alg=2,
        background_K=1,
        snr_noise_std=None,
    )
    assert torch.isfinite(torch.tensor(out["loss_util"]))
```

- [ ] **Step 6.2: Run tests — confirm `ImportError` on `stage_b_step`**

Run: `.venv/bin/pytest tests/test_trainer.py -v`

Expected: ImportError.

- [ ] **Step 6.3: Implement `stage_b_step`**

Write `privacy/trainer.py`:

```python
"""One-iteration Stage-B step. Pure function-of-modules so it's unit-testable.

The `backbone` is a `ResNet_GT` (or `ResNet_GT_phase`) instance. Stage-B contract:
  * encoder layers (conv1, bn1, layer1, layer2) — TRAIN, gradient flows from utility AND privacy terms
  * receiver layers (layer3, layer4, fc) — FROZEN, no gradient
  * adversary head — TRAIN, separately (inner loop), gradient from CE/BCE on intercepted features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import priv_loss_ce, priv_loss_entropy, priv_loss_entropy_multilabel


def _freeze_receiver(backbone):
    """Set requires_grad=False on layer3, layer4, fc; eval() on layer3/4 BN running stats."""
    for m in (backbone.layer3, backbone.layer4, backbone.fc):
        for p in m.parameters():
            p.requires_grad = False


def _flatten_imagenet_target(target_per_image: torch.Tensor) -> torch.Tensor:
    """ITIT (single-label): (B, K+1) -> (B*(K+1),) flat label per stacked image."""
    return target_per_image.reshape(-1)


def _to_khot(target_per_image: torch.Tensor, num_classes: int) -> torch.Tensor:
    """GTGT-FM (multilabel): (B, K+1) -> (B, num_classes) K-hot float tensor."""
    B, _ = target_per_image.shape
    out = torch.zeros(B, num_classes, dtype=torch.float32, device=target_per_image.device)
    out.scatter_(1, target_per_image, 1.0)
    return out


def stage_b_step(
    *,
    backbone,
    adversary,
    enc_optimizer,
    adv_optimizer,
    images: torch.Tensor,                  # (B, K+1, C, H, W)
    firearm_target: torch.Tensor,          # (B,)
    imagenet_target_per_image: torch.Tensor,  # (B, K+1)
    priv_loss_name: str,                   # "ce" or "entropy"
    lam: float,                            # privacy weight
    k_adv: int,                            # adversary inner steps
    device: torch.device,
    gt_alg: int,                           # 1 (ITIT) or 2 (GTGT-FM)
    background_K: int,                     # group_size - 1
    snr_noise_std,                         # float or None
):
    """One Stage-B iteration on a minibatch. Returns a metrics dict.

    Side effects:
      * Updates `adversary` parameters via `adv_optimizer` (k_adv steps).
      * Updates encoder params via `enc_optimizer` (1 step).
      * Receiver params are NOT updated (frozen by design).
    """
    assert priv_loss_name in ("ce", "entropy"), f"priv_loss_name must be 'ce' or 'entropy', got {priv_loss_name!r}"
    assert gt_alg in (1, 2), f"only ITIT (1) and GTGT-FM (2) are supported in Stage B; got {gt_alg}"

    _freeze_receiver(backbone)
    backbone = backbone.to(device)
    adversary = adversary.to(device)
    images = images.to(device)
    firearm_target = firearm_target.to(device)
    imagenet_target_per_image = imagenet_target_per_image.to(device)

    # --- Adversary inner loop (TTUR): k_adv steps on the same minibatch ---
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False  # freeze backbone temporarily so grads only hit adversary

    # Cache post-channel features once.
    with torch.no_grad():
        pre = backbone.encode(images)
        post_channel, _, _ = backbone.channel(pre, noise_std=snr_noise_std,
                                              gpu=device.index if device.type == "cuda" else None)

    if gt_alg == 1:
        # ITIT: per-image features. Shape after encode: (B, K+1, C', H', W').
        # But channel() collapses K -> 1 (sums and divides). For ITIT, K+1 = 1 typically; if it's >1,
        # we still treat each image's individual feature path. Re-encode without summing for the adversary.
        # Strategy: re-run encode and treat each (B, K+1, C', H', W') slice as a sample.
        pre_for_adv = pre  # (B, K+1, C', H', W')
        B, K, Cf, Hf, Wf = pre_for_adv.shape
        adv_input = pre_for_adv.reshape(B * K, Cf, Hf, Wf)
        adv_target_flat = _flatten_imagenet_target(imagenet_target_per_image)
    else:
        # GTGT-FM: adversary sees the summed post-channel feature.
        adv_input = post_channel  # (B, C', H', W')
        adv_target_khot = _to_khot(imagenet_target_per_image, num_classes=adversary.fc.out_features)

    for _ in range(k_adv):
        adv_logits = adversary(adv_input)
        if gt_alg == 1:
            loss_adv = F.cross_entropy(adv_logits, adv_target_flat)
        else:
            loss_adv = F.binary_cross_entropy_with_logits(adv_logits, adv_target_khot)
        adv_optimizer.zero_grad(set_to_none=True)
        loss_adv.backward()
        adv_optimizer.step()

    # --- Encoder outer step (1 step, fresh forward through trainable encoder, frozen receiver) ---
    backbone.train()
    # Re-enable grads only on encoder layers.
    for m in (backbone.conv1, backbone.bn1, backbone.layer1, backbone.layer2):
        for p in m.parameters():
            p.requires_grad = True

    pre = backbone.encode(images)
    post_channel, _, _ = backbone.channel(pre, noise_std=snr_noise_std,
                                          gpu=device.index if device.type == "cuda" else None)

    # Receiver forward (frozen weights, but no_grad isn't needed because we need grad to flow back into encoder).
    util_logits = backbone.decode(post_channel)
    loss_util = F.cross_entropy(util_logits, firearm_target)

    # Adversary forward for the privacy term (adversary weights NOT updated here).
    if gt_alg == 1:
        B, K, Cf, Hf, Wf = pre.shape
        adv_input2 = pre.reshape(B * K, Cf, Hf, Wf)
        adv_logits2 = adversary(adv_input2)
        if priv_loss_name == "ce":
            loss_priv = priv_loss_ce(adv_logits2, adv_target_flat)
        else:
            loss_priv = priv_loss_entropy(adv_logits2)
    else:
        adv_logits2 = adversary(post_channel)
        if priv_loss_name == "ce":
            # CE form for multilabel: -BCE
            loss_priv = -F.binary_cross_entropy_with_logits(adv_logits2, adv_target_khot)
        else:
            loss_priv = priv_loss_entropy_multilabel(adv_logits2)

    total = loss_util - lam * loss_priv  # encoder MINIMIZES this; subtract privacy term gives gradient reversal
    enc_optimizer.zero_grad(set_to_none=True)
    total.backward()
    enc_optimizer.step()

    return {
        "loss_util": loss_util.detach().item(),
        "loss_priv": loss_priv.detach().item(),
        "loss_adv": loss_adv.detach().item(),
        "loss_total": total.detach().item(),
    }
```

> **Note on subtle gradient flow:** in the encoder outer step, the adversary's parameters have `requires_grad=True` but we don't call `adv_optimizer.step()`, so they're not updated. The privacy gradient flows *through* the adversary into the encoder. That's correct — we want the encoder to receive a gradient that says "make features that make the *current* adversary uninformative". If you'd rather not waste gradient compute on adversary parameters in this step, wrap the adversary call in `for p in adversary.parameters(): p.requires_grad = False` before the encoder forward and re-enable at the start of the next iteration. The unit tests above don't depend on which strategy you pick.

- [ ] **Step 6.4: Run tests — confirm 4 PASSED**

Run: `.venv/bin/pytest tests/test_trainer.py -v`

Expected: 4 PASSED. If `test_stage_b_step_freezes_receiver` fails, audit `_freeze_receiver` — note that `requires_grad=False` is the contract, and the test verifies decoder params have `grad is None or grad == 0`.

- [ ] **Step 6.5: Commit**

```bash
git add privacy/trainer.py tests/test_trainer.py
git commit -m "feat(privacy): add stage_b_step — single iteration of Stage B

Implements the (k_adv inner + 1 outer) update:
  * Adversary inner loop: cached post-channel features, BCE/CE on labels
  * Encoder outer step: utility CE - lambda * privacy term (entropy or -CE)
  * Receiver layer3/4/fc held frozen via requires_grad=False

Unit tests verify gradient-flow contract (only encoder + adversary
update; receiver does not) and that all four loss configurations
(itit-ce, itit-entropy, gtgtfm-ce, gtgtfm-entropy) execute."
```

---

## Task 7: Stage B full training script (entry point)

**Files:**
- Create: `privacy/train_privacy.py`

This is the user-facing entry point. It loads a Stage A checkpoint, builds the privacy dataset, instantiates the adversary, and runs `stage_b_step` over epochs. Validation each epoch reports utility (firearm Acc@1) on the existing val loader.

- [ ] **Step 7.1: Implement `privacy/train_privacy.py`**

Write `privacy/train_privacy.py`:

```python
"""Stage B + Stage A_recovery entry point for privacy-preserving OTA-NGT training.

Usage:
    python -m privacy.train_privacy \
        --stage-a-ckpt Trained_Models/SmokeTest/checkpoint.pth.tar \
        --data data/GroupTestingDataset --task-num 2 --background-K 0 \
        --GT-alg 1 -a resnet18 --priv-loss entropy --lambda 1.0 --k-adv 5 \
        --stage-b-epochs 5 --recovery-epochs 1 --batch-size 16 \
        --output_dir Trained_Models/PrivacySmoke
"""
import argparse
import json
import os
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.trainer import stage_b_step


def get_parser():
    p = argparse.ArgumentParser("Privacy-preserving OTA-NGT — Stage B + recovery")
    # Core data + arch (mirror main.py where applicable)
    p.add_argument("--data", required=True, help="path to GroupTestingDataset root")
    p.add_argument("--task-num", type=int, default=2)
    p.add_argument("--background-K", type=int, required=True, help="group_size - 1; 0 for ITIT")
    p.add_argument("--GT-alg", type=int, choices=[1, 2], required=True, help="1=ITIT, 2=GTGT-FM")
    p.add_argument("-a", "--arch", required=True, help="model arch in resnet_design2")
    p.add_argument("--phase", action="store_true", help="enable random phase shift channel")
    # Channel / SNR (Stage B inherits Stage A's channel)
    p.add_argument("--SNR", type=float, default=None)
    # Stage A checkpoint
    p.add_argument("--stage-a-ckpt", required=True, help="path to checkpoint produced by main.py")
    # Stage B knobs
    p.add_argument("--priv-loss", choices=["ce", "entropy"], default="entropy")
    p.add_argument("--lambda", type=float, default=1.0, dest="lam")
    p.add_argument("--k-adv", type=int, default=5)
    p.add_argument("--stage-b-epochs", type=int, default=30)
    p.add_argument("--recovery-epochs", type=int, default=2)
    # Optimizers
    p.add_argument("--enc-lr", type=float, default=1e-3)
    p.add_argument("--adv-lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Loader
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("-j", "--workers", type=int, default=8)
    p.add_argument("-valj", "--val-workers", type=int, default=4)
    # Output
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log-name", default="privacy.log")
    p.add_argument("--print-freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    return p


def snr_to_noise_std(snr_db, signal_power=1.0):
    """Same convention as main.snr_update_function (linear power)."""
    if snr_db is None:
        return None
    return float((signal_power / (10 ** (snr_db / 10))) ** 0.5)


def build_datasets(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_list, val_list = [], []
    for folder_idx in range(args.task_num):
        train_dir = os.path.join(args.data, str(folder_idx), "train")
        val_dir = os.path.join(args.data, str(folder_idx), "val")
        train_list.append(datasets.ImageFolder(train_dir, train_t))
        val_list.append(datasets.ImageFolder(val_dir, val_t))
    return train_list, val_list


def load_stage_a(args, device):
    """Build a fresh `ResNet_GT` (matching args), load Stage-A weights, return it."""
    ctor = getattr(models, args.arch)
    model = ctor(pretrained=False, gt=True, phase=args.phase)
    ckpt = torch.load(args.stage_a_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    # Strip "module." prefix if Stage A used DataParallel.
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device)


def validate_utility(backbone, val_dataset, args, device):
    """Quick utility validation: firearm Acc@1 on a single-pass val loader."""
    backbone.eval()
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_workers, pin_memory=True, drop_last=False,
    )
    correct = total = 0
    with torch.no_grad():
        for images, firearm_target, _ in loader:
            images = images.to(device)
            firearm_target = firearm_target.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre, noise_std=snr_to_noise_std(args.SNR),
                                          gpu=device.index if device.type == "cuda" else None)
            logits = backbone.decode(post)
            pred = logits.argmax(dim=-1)
            correct += (pred == firearm_target).sum().item()
            total += firearm_target.numel()
    return correct / max(total, 1)


def stage_b_loop(backbone, adversary, train_dataset, val_dataset, args, device, log):
    enc_params = list(backbone.conv1.parameters()) + list(backbone.bn1.parameters()) \
        + list(backbone.layer1.parameters()) + list(backbone.layer2.parameters())
    enc_optim = torch.optim.SGD(enc_params, lr=args.enc_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    adv_optim = torch.optim.SGD(adversary.parameters(), lr=args.adv_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    snr_noise = snr_to_noise_std(args.SNR)

    for epoch in range(args.stage_b_epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        t0 = time.time()
        for it, (images, firearm_target, imagenet_target_per_image) in enumerate(loader):
            metrics = stage_b_step(
                backbone=backbone, adversary=adversary,
                enc_optimizer=enc_optim, adv_optimizer=adv_optim,
                images=images, firearm_target=firearm_target,
                imagenet_target_per_image=imagenet_target_per_image,
                priv_loss_name=args.priv_loss, lam=args.lam, k_adv=args.k_adv,
                device=device, gt_alg=args.GT_alg, background_K=args.background_K,
                snr_noise_std=snr_noise,
            )
            if it % args.print_freq == 0:
                line = (f"[StageB][ep {epoch}][it {it:5d}] "
                        f"util={metrics['loss_util']:.4f} priv={metrics['loss_priv']:.4f} "
                        f"adv={metrics['loss_adv']:.4f} total={metrics['loss_total']:.4f}")
                print(line); log.write(line + "\n"); log.flush()
        acc = validate_utility(backbone, val_dataset, args, device)
        line = f"[StageB][ep {epoch}] val_firearm_acc={acc:.4f} time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def stage_a_recovery_loop(backbone, train_dataset, val_dataset, args, device, log):
    """Brief unfreeze-and-train-on-utility-only pass to recover utility regression from Stage B."""
    for p in backbone.parameters():
        p.requires_grad = True
    backbone.train()
    optim = torch.optim.SGD(backbone.parameters(), lr=args.enc_lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
    snr_noise = snr_to_noise_std(args.SNR)
    for epoch in range(args.recovery_epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        t0 = time.time()
        for it, (images, firearm_target, _) in enumerate(loader):
            images = images.to(device)
            firearm_target = firearm_target.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                          gpu=device.index if device.type == "cuda" else None)
            logits = backbone.decode(post)
            loss = F.cross_entropy(logits, firearm_target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if it % args.print_freq == 0:
                line = f"[Recovery][ep {epoch}][it {it:5d}] util={loss.item():.4f}"
                print(line); log.write(line + "\n"); log.flush()
        acc = validate_utility(backbone, val_dataset, args, device)
        line = f"[Recovery][ep {epoch}] val_firearm_acc={acc:.4f} time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def save_ckpt(backbone, adversary, args, path):
    torch.save({
        "state_dict_backbone": backbone.state_dict(),
        "state_dict_adversary": adversary.state_dict(),
        "args": vars(args),
    }, path)


def main():
    args = get_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_name)
    log = open(log_path, "w")
    log.write(f"args: {json.dumps(vars(args), indent=2)}\n"); log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbone = load_stage_a(args, device)
    train_list, val_list = build_datasets(args)
    train_dataset = PrivacyTaskCoalitionDataset(train_list, args, split="train")
    val_dataset = PrivacyTaskCoalitionDataset(val_list, args, split="val")

    adversary = AdversaryHead(arch_name=args.arch, num_classes=train_dataset.num_imagenet_classes).to(device)
    print(f"Adversary num_classes={train_dataset.num_imagenet_classes}")

    stage_b_loop(backbone, adversary, train_dataset, val_dataset, args, device, log)
    save_ckpt(backbone, adversary, args, os.path.join(args.output_dir, "stage_b_final.pth.tar"))

    if args.recovery_epochs > 0:
        stage_a_recovery_loop(backbone, train_dataset, val_dataset, args, device, log)
        save_ckpt(backbone, adversary, args, os.path.join(args.output_dir, "stage_b_recovered.pth.tar"))

    log.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Add a smoke test that runs 2 iterations end-to-end on real data**

Write `tests/test_smoke_train.py`:

```python
"""End-to-end smoke test for privacy.train_privacy.

Marked slow + gpu — runs only if the dataset and a CUDA device are present.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "GroupTestingDataset"
STAGE_A_CKPT = REPO_ROOT / "Trained_Models" / "SmokeTest" / "checkpoint.pth.tar"


@pytest.mark.slow
@pytest.mark.gpu
def test_train_privacy_smoke(tmp_path):
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

    out_dir = tmp_path / "PrivacySmoke"
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"), "-u", "-m", "privacy.train_privacy",
        "--stage-a-ckpt", str(STAGE_A_CKPT),
        "--data", str(DATA_ROOT), "--task-num", "2", "--background-K", "0",
        "--GT-alg", "1", "-a", "resnet18", "--priv-loss", "entropy",
        "--lambda", "1.0", "--k-adv", "2",
        "--stage-b-epochs", "1", "--recovery-epochs", "0",
        "--batch-size", "8", "-j", "2", "-valj", "1", "--print-freq", "1",
        "--output_dir", str(out_dir), "--log-name", "smoke.log",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"train_privacy failed:\n{proc.stdout}\n{proc.stderr}"
    assert (out_dir / "stage_b_final.pth.tar").exists()
```

- [ ] **Step 7.3: Run unit tests (no Stage-A ckpt required); they should still all pass**

Run: `.venv/bin/pytest tests/ -v -m "not slow"`

Expected: all previously-passing tests (Tasks 1-6) still pass. The smoke test from this step is marked `slow` and is therefore skipped here.

- [ ] **Step 7.4: Run the smoke test if the smoke-test checkpoint exists**

Run: `.venv/bin/pytest tests/test_smoke_train.py -v -m "slow and gpu"`

Expected: PASS in <15 minutes on the L40 if `Trained_Models/SmokeTest/checkpoint.pth.tar` is present from the earlier smoke test. SKIPPED otherwise (and that's fine — the unit tests cover the logic).

- [ ] **Step 7.5: Commit**

```bash
git add privacy/train_privacy.py tests/test_smoke_train.py
git commit -m "feat(privacy): add Stage B + recovery training script + smoke test

Loads a Stage A checkpoint, runs --stage-b-epochs of Stage B (frozen
receiver, k_adv adversary inner loop, encoder utility-minus-lambda-priv
outer step), then optional --recovery-epochs of utility-only recovery.
Saves stage_b_final.pth.tar (always) and stage_b_recovered.pth.tar
(if recovery enabled).

Smoke test runs one short epoch end-to-end on the existing dataset and
smoke-test Stage A checkpoint; marked slow + gpu so the unit-test
sweep stays fast."
```

---

## Task 8: Stage C eval script (honest leakage measurement)

**Files:**
- Create: `privacy/eval_privacy.py`
- Create: `tests/test_smoke_eval.py`

- [ ] **Step 8.1: Implement `privacy/eval_privacy.py`**

Write `privacy/eval_privacy.py`:

```python
"""Stage C: train a fresh-init adversary from scratch on a frozen privacy-fine-tuned encoder.

Reports its top-1 ImageNet accuracy (ITIT) or per-class mean ROC-AUC + mAP (GTGT-FM)
as the honest leakage measurement.

Usage:
    python -m privacy.eval_privacy \
        --stage-b-ckpt Trained_Models/PrivacySmoke/stage_b_final.pth.tar \
        --data data/GroupTestingDataset --task-num 2 --background-K 0 \
        --GT-alg 1 -a resnet18 --stage-c-epochs 5 --batch-size 16 \
        --output_dir Trained_Models/PrivacySmoke/EvalC
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset


def get_parser():
    p = argparse.ArgumentParser("Privacy-preserving OTA-NGT — Stage C honest leakage eval")
    p.add_argument("--data", required=True)
    p.add_argument("--task-num", type=int, default=2)
    p.add_argument("--background-K", type=int, required=True)
    p.add_argument("--GT-alg", type=int, choices=[1, 2], required=True)
    p.add_argument("-a", "--arch", required=True)
    p.add_argument("--phase", action="store_true")
    p.add_argument("--SNR", type=float, default=None)
    p.add_argument("--stage-b-ckpt", required=True)
    p.add_argument("--stage-c-epochs", type=int, default=60)
    p.add_argument("--adv-lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("-j", "--workers", type=int, default=8)
    p.add_argument("-valj", "--val-workers", type=int, default=4)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log-name", default="eval_c.log")
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    return p


def snr_to_noise_std(snr_db, signal_power=1.0):
    if snr_db is None:
        return None
    return float((signal_power / (10 ** (snr_db / 10))) ** 0.5)


def build_datasets(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_list, val_list = [], []
    for folder_idx in range(args.task_num):
        train_list.append(datasets.ImageFolder(os.path.join(args.data, str(folder_idx), "train"), train_t))
        val_list.append(datasets.ImageFolder(os.path.join(args.data, str(folder_idx), "val"), val_t))
    return train_list, val_list


def load_backbone(args, device):
    ctor = getattr(models, args.arch)
    backbone = ctor(pretrained=False, gt=True, phase=args.phase)
    ckpt = torch.load(args.stage_b_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict_backbone"] if "state_dict_backbone" in ckpt else ckpt["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    backbone.load_state_dict(state)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def train_fresh_adversary(backbone, adversary, train_dataset, args, device, log):
    snr_noise = snr_to_noise_std(args.SNR)
    optim = torch.optim.SGD(adversary.parameters(), lr=args.adv_lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
    for epoch in range(args.stage_c_epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        adversary.train()
        t0 = time.time()
        for it, (images, _, imagenet_target_per_image) in enumerate(loader):
            images = images.to(device); imagenet_target_per_image = imagenet_target_per_image.to(device)
            with torch.no_grad():
                pre = backbone.encode(images)
                post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                              gpu=device.index if device.type == "cuda" else None)
            if args.GT_alg == 1:
                B, K, Cf, Hf, Wf = pre.shape
                adv_in = pre.reshape(B * K, Cf, Hf, Wf)
                adv_target = imagenet_target_per_image.reshape(-1)
                logits = adversary(adv_in)
                loss = F.cross_entropy(logits, adv_target)
            else:
                num_classes = adversary.fc.out_features
                khot = torch.zeros(images.size(0), num_classes, device=device).scatter_(
                    1, imagenet_target_per_image, 1.0)
                logits = adversary(post)
                loss = F.binary_cross_entropy_with_logits(logits, khot)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            if it % args.print_freq == 0:
                line = f"[StageC][ep {epoch}][it {it:5d}] adv_loss={loss.item():.4f}"
                print(line); log.write(line + "\n"); log.flush()
        line = f"[StageC][ep {epoch}] time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def evaluate_leakage(backbone, adversary, val_dataset, args, device):
    """Returns dict with leakage metrics on val set."""
    snr_noise = snr_to_noise_std(args.SNR)
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_workers, pin_memory=True, drop_last=False,
    )
    adversary.eval()
    correct = total = 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, _, imagenet_target_per_image in loader:
            images = images.to(device); imagenet_target_per_image = imagenet_target_per_image.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                          gpu=device.index if device.type == "cuda" else None)
            if args.GT_alg == 1:
                B, K, Cf, Hf, Wf = pre.shape
                adv_in = pre.reshape(B * K, Cf, Hf, Wf)
                adv_target = imagenet_target_per_image.reshape(-1)
                logits = adversary(adv_in)
                pred = logits.argmax(dim=-1)
                correct += (pred == adv_target).sum().item()
                total += adv_target.numel()
            else:
                logits = adversary(post)
                all_logits.append(logits.cpu().numpy())
                num_classes = adversary.fc.out_features
                khot = torch.zeros(images.size(0), num_classes, device=device).scatter_(
                    1, imagenet_target_per_image, 1.0)
                all_targets.append(khot.cpu().numpy())

    if args.GT_alg == 1:
        return {"top1_imagenet_acc": correct / max(total, 1)}

    # GTGT-FM: per-class AUC + mAP
    from sklearn.metrics import roc_auc_score, average_precision_score
    logits = np.concatenate(all_logits, axis=0); targets = np.concatenate(all_targets, axis=0)
    aucs, aps = [], []
    for k in range(targets.shape[1]):
        if targets[:, k].sum() == 0:
            continue  # class never appeared in val
        aucs.append(roc_auc_score(targets[:, k], logits[:, k]))
        aps.append(average_precision_score(targets[:, k], logits[:, k]))
    return {"mean_auc": float(np.mean(aucs)), "mean_ap": float(np.mean(aps)),
            "num_classes_evaluated": len(aucs)}


def main():
    args = get_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log = open(os.path.join(args.output_dir, args.log_name), "w")
    log.write(f"args: {json.dumps(vars(args), indent=2)}\n"); log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone(args, device)
    train_list, val_list = build_datasets(args)
    train_dataset = PrivacyTaskCoalitionDataset(train_list, args, split="train")
    val_dataset = PrivacyTaskCoalitionDataset(val_list, args, split="val")

    adversary = AdversaryHead(arch_name=args.arch, num_classes=train_dataset.num_imagenet_classes).to(device)

    train_fresh_adversary(backbone, adversary, train_dataset, args, device, log)

    metrics = evaluate_leakage(backbone, adversary, val_dataset, args, device)
    print("Leakage metrics:", json.dumps(metrics, indent=2))
    log.write("leakage: " + json.dumps(metrics) + "\n"); log.flush()
    with open(os.path.join(args.output_dir, "leakage.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Smoke test for Stage C**

Write `tests/test_smoke_eval.py`:

```python
"""End-to-end smoke test for privacy.eval_privacy.

Requires a Stage-B checkpoint (produced by privacy.train_privacy).
Marked slow + gpu.
"""
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "GroupTestingDataset"


@pytest.mark.slow
@pytest.mark.gpu
def test_eval_privacy_smoke(tmp_path):
    if not DATA_ROOT.exists():
        pytest.skip(f"Missing dataset at {DATA_ROOT}")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not installed")

    # Find any Stage-B checkpoint we can use.
    ckpts = list(REPO_ROOT.glob("Trained_Models/**/stage_b_final.pth.tar"))
    if not ckpts:
        pytest.skip("No stage_b_final.pth.tar checkpoint in Trained_Models/; run privacy.train_privacy first")
    ckpt = ckpts[0]

    out_dir = tmp_path / "EvalC"
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"), "-u", "-m", "privacy.eval_privacy",
        "--stage-b-ckpt", str(ckpt),
        "--data", str(DATA_ROOT), "--task-num", "2", "--background-K", "0",
        "--GT-alg", "1", "-a", "resnet18",
        "--stage-c-epochs", "1", "--batch-size", "8",
        "-j", "2", "-valj", "1", "--print-freq", "5",
        "--output_dir", str(out_dir),
    ]
    env = os.environ.copy(); env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"eval_privacy failed:\n{proc.stdout}\n{proc.stderr}"
    leakage_path = out_dir / "leakage.json"
    assert leakage_path.exists()
    import json
    with open(leakage_path) as f:
        m = json.load(f)
    assert "top1_imagenet_acc" in m
```

- [ ] **Step 8.3: Verify all unit tests still pass**

Run: `.venv/bin/pytest tests/ -v -m "not slow"`

Expected: every test from Tasks 1-7 still passes. No regressions.

- [ ] **Step 8.4: Commit**

```bash
git add privacy/eval_privacy.py tests/test_smoke_eval.py
git commit -m "feat(privacy): add Stage C honest leakage eval script

Loads a Stage-B checkpoint, freezes the encoder, trains a fresh-init
adversary from scratch for --stage-c-epochs, then reports top-1 ImageNet
accuracy (ITIT) or per-class mean AUC/mAP (GTGT-FM) on the val set.
Output goes to <output_dir>/leakage.json so trade-off curves can be
compiled across runs by simple json reads."
```

---

## Task 9: Update README with privacy training section

**Files:**
- Modify: `README.md`

- [ ] **Step 9.1: Add a new section after the existing "Example Usage" section**

Open `README.md`. After the existing Example Usage section (the four `--multiprocessing-distributed` blocks ending around line 124), and BEFORE the `## Dataset Preparation` section, insert:

````markdown
---

## Privacy-Preserving Training (Stage A → B → C)

The `privacy/` package adds a privacy-preserving variant in which the encoder is fine-tuned so that its post-channel features remain useful for binary firearm detection but reveal little about the input image's fine-grained ImageNet class.

### Stage A — utility pretrain (existing)

Use `main.py` exactly as in the examples above to produce a Stage A checkpoint. Both supported algorithms are ITIT (`--GT-alg 1`) and GTGT-FM (`--GT-alg 2`).

### Stage B — privacy fine-tune (frozen receiver)

```bash
.venv/bin/python -u -m privacy.train_privacy \
    --stage-a-ckpt Trained_Models/StageA/checkpoint.pth.tar \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 \
    --GT-alg 1 -a resnext101_32x8d \
    --priv-loss entropy --lambda 1.0 --k-adv 5 \
    --stage-b-epochs 30 --recovery-epochs 2 \
    --batch-size 32 -j 8 -valj 4 \
    --output_dir Trained_Models/Privacy/lambda_1.0_entropy
```

Key flags:
- `--priv-loss {ce,entropy}` — privacy term form. `entropy` (negative entropy of adversary's softmax) is the principled default; `ce` (negated CE) is the DANN-style ablation.
- `--lambda` — weight of the privacy term. Sweep `{0, 0.1, 0.3, 1.0, 3.0, 10.0}` per run.
- `--k-adv` — adversary inner-loop steps per encoder step (TTUR).
- `--stage-b-epochs` / `--recovery-epochs` — Stage B and the post-Stage-B utility recovery, respectively.

For GTGT-FM, set `--GT-alg 2 --background-K 7` (group size 8), and the adversary automatically becomes multilabel (1000-way sigmoid + BCE on the K-hot label vector).

### Stage C — honest leakage eval

```bash
.venv/bin/python -u -m privacy.eval_privacy \
    --stage-b-ckpt Trained_Models/Privacy/lambda_1.0_entropy/stage_b_final.pth.tar \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 \
    --GT-alg 1 -a resnext101_32x8d \
    --stage-c-epochs 60 --batch-size 32 -j 8 -valj 4 \
    --output_dir Trained_Models/Privacy/lambda_1.0_entropy/EvalC
```

Reports leakage in `leakage.json`:
- ITIT: `{"top1_imagenet_acc": ...}`
- GTGT-FM: `{"mean_auc": ..., "mean_ap": ..., "num_classes_evaluated": ...}`

### Tests

```bash
.venv/bin/pytest tests/ -v -m "not slow"     # unit tests (~seconds)
.venv/bin/pytest tests/ -v -m "slow and gpu" # smoke tests (~minutes; needs dataset + GPU)
```

See `docs/superpowers/specs/2026-04-17-privacy-preserving-ota-ngt-design.md` for the full design rationale.
````

- [ ] **Step 9.2: Commit**

```bash
git add README.md
git commit -m "docs: add Privacy-Preserving Training section to README

Walks through Stage A -> B -> C with concrete command lines for both
ITIT and GTGT-FM, plus the test-running cheat sheet."
```

---

## Task 10: Final integration check + push

- [ ] **Step 10.1: Run the full unit-test sweep**

Run: `.venv/bin/pytest tests/ -v -m "not slow"`

Expected: all unit tests pass. The total suite should be on the order of 25 tests.

- [ ] **Step 10.2: Run unit tests + slow smoke if dataset and GPU are present**

Run: `.venv/bin/pytest tests/ -v`

Expected: unit tests PASS; smoke tests PASS (if dataset and Stage-A checkpoint present) or SKIP (if not). No FAIL.

- [ ] **Step 10.3: Confirm `git status` is clean and no excluded artifact crept in**

Run: `git status` and `git ls-files | xargs -I{} git check-ignore {} 2>/dev/null | head`

Expected: working tree clean. No tracked file is currently ignored (the `check-ignore` pipe should print nothing). If any `.log`, `.pth.tar`, `.pkl`, or `.npy` shows up in `git status` as tracked, that's a bug — investigate and `git rm --cached` it before pushing.

- [ ] **Step 10.4: Push**

Run: `git push origin dev/Privacy_Preserving_OTA_NGT`

Expected: clean fast-forward push to the existing remote branch (set up earlier).

---

## Out of scope (future plans)

The following were called out in the spec's Section 9 and intentionally are NOT covered by this plan. Each would be a separate spec + plan:

- Differential-privacy-style noise budgets / formal DP guarantees on the encoder.
- Defenses against an adversary with paired plaintext/feature samples.
- Multi-class group testing (more than firearm vs background).
- Backbones other than ResNeXt-101 / ResNet-18 — same code should work via `--arch` but not validated for this paper.
- ResNet_GT_phase + privacy training combination (the channel split is in place via Task 2, but no smoke test exercises it; expect minor adjustments in `stage_b_step` for complex tensors if you enable `--phase`).
