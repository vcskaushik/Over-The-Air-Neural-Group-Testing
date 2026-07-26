#!/bin/bash
# Disk-quota-safe extraction of the already-downloaded train parts (no assembly).
# The full 138G tar is NEVER materialized. Phase 1 streams the parts in order into
# the outer tar extractor, deleting each part right after it is piped, so
# parts+inner-tars stay ~138G (a 1:1 swap) instead of doubling. Phase 2 extracts
# the 1000 inner <wnid>.tar in parallel, deleting each as it goes.
set -u
cd "$(dirname "$0")"
ROOT=data/ImageNet-ILSVRC2012
PD=data/ILSVRC2012_img_train.tar.parts
STAGE=data/train_tars
mkdir -p "$STAGE" "$ROOT/train"

np=$(ls "$PD"/part_* 2>/dev/null | wc -l)
echo "[xtr] $(date '+%T') phase1: stream $np parts -> inner tars (deleting parts as consumed)"
( for p in "$PD"/part_*; do cat "$p" && rm -f "$p"; done ) | tar xf - -C "$STAGE"
rc=${PIPESTATUS[1]}
ni=$(ls "$STAGE"/*.tar 2>/dev/null | wc -l)
echo "[xtr] $(date '+%T') phase1 rc=$rc inner_tars=$ni parts_left=$(ls "$PD"/part_* 2>/dev/null | wc -l)"
[ "$rc" -ne 0 ] && { echo "[xtr] outer extract FAILED (rc=$rc)"; exit 1; }
rmdir "$PD" 2>/dev/null || true

echo "[xtr] $(date '+%T') phase2: parallel-extract $ni inner tars -> train/<wnid> (deleting as consumed)"
ls "$STAGE"/*.tar | xargs -P 16 -I{} sh -c '
  w=$(basename "{}" .tar); d="'"$ROOT"'/train/$w"
  mkdir -p "$d" && tar xf "{}" -C "$d" && rm -f "{}"
'
left=$(ls "$STAGE"/*.tar 2>/dev/null | wc -l)
n=$(find "$ROOT/train" -type f 2>/dev/null | wc -l); c=$(ls "$ROOT/train" 2>/dev/null | wc -l)
echo "[xtr] $(date '+%T') train files=$n classes=$c inner_left=$left"
rmdir "$STAGE" 2>/dev/null || true
[ "$c" -ge 1000 ] && [ "$n" -ge 1230000 ] && echo "TRAIN_READY" || { echo "[xtr] INCOMPLETE (classes=$c files=$n)"; exit 1; }
