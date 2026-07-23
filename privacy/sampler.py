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
