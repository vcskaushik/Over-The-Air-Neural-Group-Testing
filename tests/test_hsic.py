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
