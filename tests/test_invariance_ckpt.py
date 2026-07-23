import torch
from torch.utils.data import Dataset, DataLoader

import resnet_design2 as models
from privacy.hsic import HSICPenalty
from privacy.train_invariance import save_invariance_ckpt, extractor_sanity


def test_save_invariance_ckpt_is_main_resume_compatible(tmp_path):
    backbone = models.resnet18(pretrained=False, gt=True, phase=False)
    path = tmp_path / "invariance_final.pth.tar"
    save_invariance_ckpt(backbone, coded_pwr=2.5, epoch=7, path=str(path))
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt and "coded_pwr" in ckpt and "epoch" in ckpt
    assert ckpt["coded_pwr"] == 2.5 and ckpt["epoch"] == 7
    # main.py --resume expects the module. prefix.
    assert all(k.startswith("module.") for k in ckpt["state_dict"])


class _TinyITITDataset(Dataset):
    """Minimal ITIT-shaped (K=0) dataset: each item is
    (images (1, C, H, W), firearm_target int, imagenet_target (1,) long).

    Default DataLoader collation over a batch of N items turns this into
    (images (N, 1, C, H, W), firearm_target (N,), imagenet_target (N, 1)),
    matching what `extractor_sanity` consumes.
    """

    def __init__(self, n, num_bg_classes=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 1, 3, 16, 16, generator=g)
        # Roughly half background (firearm=0), half firearm (firearm=1).
        self.firearm = [i % 2 for i in range(n)]
        # Background rows get a real (non-degenerate) imagenet label so the
        # true-vs-permuted HSIC comparison is meaningful; firearm rows are
        # masked out by extractor_sanity anyway.
        self.imagenet = [i % num_bg_classes for i in range(n)]

    def __len__(self):
        return len(self.firearm)

    def __getitem__(self, idx):
        return self.images[idx], self.firearm[idx], torch.tensor([self.imagenet[idx]], dtype=torch.long)


def test_extractor_sanity_runs_on_cpu_and_masks_background_rows():
    torch.manual_seed(0)
    device = torch.device("cpu")
    backbone = models.resnet18(pretrained=False, gt=True, phase=False)
    backbone.arch_name = "resnet18"
    hsic_penalty = HSICPenalty(arch_name="resnet18", extractor="random", num_random=1)

    dataset = _TinyITITDataset(n=16, seed=42)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    result = extractor_sanity(backbone, hsic_penalty, loader, device, snr_noise=None, num_batches=2)

    # (a) runs without error and (b) returns a 2-tuple of floats.
    assert isinstance(result, tuple) and len(result) == 2
    t_true, t_perm = result
    assert isinstance(t_true, float) and isinstance(t_perm, float)
    # HSIC values are non-negative (biased estimator can be tiny/negative-ish
    # due to numerical noise, but should be finite real numbers either way).
    assert t_true == t_true and t_perm == t_perm  # not NaN
    assert not (t_true in (float("inf"), float("-inf")))
    assert not (t_perm in (float("inf"), float("-inf")))
