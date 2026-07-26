#!/bin/bash
# Fetch ImageNet train directly from image-net.org (no auth) and extract into
# data/ImageNet-ILSVRC2012/train/<wnid>/*.JPEG. The train tar is a tar-of-tars
# (1000 inner <wnid>.tar). Merges with whatever the background sync already wrote
# (identical bytes; skip-if-exists is fine). Resumable download.
set -u
cd "$(dirname "$0")"
TAR=data/ILSVRC2012_img_train.tar
URL=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
ROOT=data/ImageNet-ILSVRC2012
STAGE=data/train_tars

echo "[train] $(date '+%T') downloading (~138G, parallel/resumable)"
bash pget.sh "$URL" "$TAR" 24 400 || { echo "[train] download FAILED"; exit 1; }
sz=$(du -h "$TAR" | cut -f1); echo "[train] $(date '+%T') downloaded $sz; extracting outer tar"

mkdir -p "$STAGE" "$ROOT/train"
tar xf "$TAR" -C "$STAGE" || { echo "[train] outer extract FAILED"; exit 1; }
ntar=$(ls "$STAGE"/*.tar 2>/dev/null | wc -l)
echo "[train] $(date '+%T') outer done: $ntar inner tars; extracting (parallel)"

ls "$STAGE"/*.tar | xargs -P 16 -I{} sh -c '
  w=$(basename "{}" .tar)
  mkdir -p "'"$ROOT"'/train/$w"
  tar xf "{}" -C "'"$ROOT"'/train/$w" && rm -f "{}"
'
left=$(ls "$STAGE"/*.tar 2>/dev/null | wc -l)
n=$(find "$ROOT/train" -type f 2>/dev/null | wc -l)
c=$(ls "$ROOT/train" 2>/dev/null | wc -l)
echo "[train] $(date '+%T') train files=$n classes=$c (inner tars left=$left)"
rm -f "$TAR"; rmdir "$STAGE" 2>/dev/null || true
[ "$c" -ge 1000 ] && [ "$n" -ge 1230000 ] && echo "TRAIN_READY" || { echo "[train] INCOMPLETE (classes=$c files=$n)"; exit 1; }
