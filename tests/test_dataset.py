"""Tests for PrivacyTaskCoalitionDataset — must surface ImageNet labels alongside binary targets."""
import os
import shutil
from argparse import Namespace
from pathlib import Path
import types

import pytest
import numpy as np
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


def test_dataset_uses_provided_wnid_mapping(synthetic_dataset):
    """When a pre-built wnid_to_imagenet_idx is supplied, the dataset must use it directly."""
    args = _build_args(synthetic_dataset, background_K=0)
    dl = _build_dataset_list(synthetic_dataset)

    # Collect all wnids that exist in the dataset list.
    all_wnids = sorted({wnid for ds in dl for wnid in ds.class_to_idx.keys()})
    # Build a shifted mapping so indices start at 100 instead of 0.
    shifted_mapping = {wnid: i + 100 for i, wnid in enumerate(all_wnids)}

    ds = PrivacyTaskCoalitionDataset(dl, args, split="train",
                                     wnid_to_imagenet_idx=shifted_mapping)
    assert ds.num_imagenet_classes == len(shifted_mapping)
    # Every item's imagenet label must fall in the shifted range.
    _, _, imagenet_targets = ds[0]
    assert imagenet_targets[0].item() >= 100


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
