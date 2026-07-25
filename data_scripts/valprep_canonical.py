"""
Build canonical ImageNet val tree from the official ILSVRC2012_img_val.tar.

The pipeline (main.py GroupTestDataset_val + constants.firearm_file_paths) requires
the ORIGINAL ImageNet val filenames (ILSVRC2012_val_00000001.JPEG ..). The HF parquet
mirror dropped those names, so we materialize val from the official tar and sort each
image into its <wnid>/ folder using the ILSVRC2012 validation ground truth
(data/val_ground_truth_wnids.txt: line i -> wnid of ILSVRC2012_val_{i+1:08d}.JPEG).

Writes into data/ImageNet-ILSVRC2012/val_canonical/<wnid>/ILSVRC2012_val_*.JPEG,
then the caller atomically swaps it in for .../val.

Run from repo root:
    .venv/bin/python data_scripts/valprep_canonical.py
"""
import os, tarfile

TAR = "data/ILSVRC2012_img_val.tar"
GT = "data/val_ground_truth_wnids.txt"
OUT = "data/ImageNet-ILSVRC2012/val_canonical"


def main():
    wnids = open(GT).read().split()
    assert len(wnids) == 50000, len(wnids)
    os.makedirs(OUT, exist_ok=True)
    for w in set(wnids):
        os.makedirs(os.path.join(OUT, w), exist_ok=True)

    n = 0
    with tarfile.open(TAR, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)  # ILSVRC2012_val_00000001.JPEG
            # index from filename
            stem = base.replace("ILSVRC2012_val_", "").replace(".JPEG", "")
            idx = int(stem)  # 1..50000
            wnid = wnids[idx - 1]
            f = tf.extractfile(m)
            data = f.read()
            with open(os.path.join(OUT, wnid, base), "wb") as out:
                out.write(data)
            n += 1
            if n % 5000 == 0:
                print(f"  {n}/50000", flush=True)
    print(f"DONE {n} val images -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
