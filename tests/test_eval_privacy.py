import types

import torch

from privacy.eval_privacy import BackgroundValDataset, build_background_val_index


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


def test_background_val_dataset_shapes_and_labels():
    index = [("/d/n01/x.jpg", 1), ("/d/n02/z.jpg", 2)]
    loader = lambda p: torch.zeros(3, 8, 8)
    transform = lambda x: x

    ds = BackgroundValDataset(index, loader, transform)
    assert len(ds) == 2

    images, firearm_target, imagenet_targets = ds[0]
    assert images.shape == (1, 3, 8, 8)
    assert firearm_target == 0
    assert imagenet_targets.shape == (1,)
    assert imagenet_targets[0].item() == 1

    images, firearm_target, imagenet_targets = ds[1]
    assert images.shape == (1, 3, 8, 8)
    assert firearm_target == 0
    assert imagenet_targets[0].item() == 2
