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
