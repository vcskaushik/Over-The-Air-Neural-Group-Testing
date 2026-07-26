#!/bin/bash
# Fetch ImageNet val directly from image-net.org (no auth), build the canonical
# per-wnid val tree, and swap it into data/ImageNet-ILSVRC2012/val (where the
# GroupTestingDataset val symlinks point). Resumable download.
set -u
cd "$(dirname "$0")"
TAR=data/ILSVRC2012_img_val.tar
URL=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
ROOT=data/ImageNet-ILSVRC2012

echo "[val] $(date '+%T') downloading (6.3G, parallel/resumable)"
bash pget.sh "$URL" "$TAR" 16 400 || { echo "[val] download FAILED"; exit 1; }
sz=$(du -h "$TAR" | cut -f1); echo "[val] $(date '+%T') downloaded $sz"

echo "[val] $(date '+%T') valprep -> val_canonical"
rm -rf "$ROOT/val_canonical"
.venv/bin/python data_scripts/valprep_canonical.py || { echo "[val] valprep FAILED"; exit 1; }

echo "[val] $(date '+%T') swapping val_canonical -> val"
rm -rf "$ROOT/val"
mv "$ROOT/val_canonical" "$ROOT/val"
n=$(find "$ROOT/val" -type f 2>/dev/null | wc -l)
c=$(ls "$ROOT/val" 2>/dev/null | wc -l)
echo "[val] $(date '+%T') val files=$n classes=$c"
rm -f "$TAR"
[ "$n" -ge 49000 ] && echo "VAL_READY" || { echo "[val] INCOMPLETE ($n<49000)"; exit 1; }
