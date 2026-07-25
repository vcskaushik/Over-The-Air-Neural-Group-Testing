"""
Materialize ImageNet-1k from public HF parquet mirrors (mrm8488/ImageNet1K-{train,val})
into the per-wnid ImageFolder layout the OTA-NGT pipeline expects:

    data/ImageNet-ILSVRC2012/{train,val}/<wnid>/*.JPEG

The parquet 'image' column holds original JPEG bytes, so we write them out verbatim
(no re-encode). 'label' is the canonical ImageNet class index (0..999), verified to
match torchvision ordering; index -> wnid via data/imagenet_class_index.json.

Streams shard-by-shard (download -> extract -> delete parquet) to bound disk usage,
and is resumable via per-shard .done markers.

Run from repo root:
    .venv/bin/python data_scripts/materialize_imagenet_from_hf.py train --workers 8
    .venv/bin/python data_scripts/materialize_imagenet_from_hf.py val   --workers 8
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO = {"train": "mrm8488/ImageNet1K-train", "val": "mrm8488/ImageNet1K-val"}
OUT_ROOT = "data/ImageNet-ILSVRC2012"
CACHE = "data/hf_cache"
MARK_DIR = "data/.materialize_markers"

CLASS_INDEX = json.load(open("data/imagenet_class_index.json"))
IDX2WNID = {int(k): v[0] for k, v in CLASS_INDEX.items()}


def list_parquets(repo_id):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    pq = sorted(f for f in files if f.endswith(".parquet"))
    return pq


def process_shard(split, shard_idx, filename):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo_id = REPO[split]
    mark = os.path.join(MARK_DIR, f"{split}-{shard_idx:05d}.done")
    if os.path.exists(mark):
        return (shard_idx, "skip", 0)

    path = hf_hub_download(repo_id=repo_id, filename=filename,
                           repo_type="dataset", cache_dir=CACHE)
    out_base = os.path.join(OUT_ROOT, split)
    n = 0
    pf = pq.ParquetFile(path)
    row0 = 0
    for batch in pf.iter_batches(batch_size=1024, columns=["image", "label"]):
        imgs = batch.column("image").to_pylist()
        labs = batch.column("label").to_pylist()
        for j, (cell, lab) in enumerate(zip(imgs, labs)):
            wnid = IDX2WNID[int(lab)]
            d = os.path.join(out_base, wnid)
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, f"{split}_{shard_idx:05d}_{row0 + j:06d}.JPEG")
            with open(fn, "wb") as fh:
                fh.write(cell["bytes"])
            n += 1
        row0 += len(labs)

    # free disk: drop the parquet blob (resolve symlink target too)
    try:
        real = os.path.realpath(path)
        os.remove(path)
        if os.path.exists(real):
            os.remove(real)
    except OSError:
        pass
    os.makedirs(MARK_DIR, exist_ok=True)
    open(mark, "w").write(str(n))
    return (shard_idx, "done", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(MARK_DIR, exist_ok=True)
    files = list_parquets(REPO[args.split])
    print(f"[{args.split}] {len(files)} parquet shards", flush=True)

    t0 = time.time()
    total = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, args.split, i, f): i
                for i, f in enumerate(files)}
        for fut in as_completed(futs):
            shard_idx, status, n = fut.result()
            total += n
            done += 1
            el = time.time() - t0
            print(f"[{args.split}] shard {shard_idx:3d} {status:4s} "
                  f"imgs={n:5d} | {done}/{len(files)} shards | "
                  f"total={total} | {el:6.0f}s", flush=True)
    print(f"[{args.split}] DONE {total} images from {len(files)} shards "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
