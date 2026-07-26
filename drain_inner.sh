#!/bin/bash
# Concurrent consumer for phase-1's inner tars (overlaps extract_train.sh phase 1
# to ~halve wall-clock and keep the inner-tar staging small for disk-quota safety).
# Phase 1 writes inner <wnid>.tar sequentially, so while it runs, every inner tar
# EXCEPT the newest (possibly mid-write) is complete -> safe to extract. Once
# phase 1's process is gone, all remaining inner tars are complete. Extracts in
# parallel with --no-same-owner (mfs forbids chown; without this tar exits non-zero
# and the `&& rm` would not fire). Exits TRAIN_READY when everything is extracted.
set -u
cd "$(dirname "$0")"
ROOT=data/ImageNet-ILSVRC2012
STAGE=data/train_tars

extract_one() {
  w=$(basename "$1" .tar); d="$ROOT/train/$w"
  mkdir -p "$d" && tar --no-same-owner -xf "$1" -C "$d" && rm -f "$1"
}
export -f extract_one; export ROOT

while true; do
  alive=$(pgrep -fc extract_train.sh 2>/dev/null || echo 0)
  if [ "$alive" -eq 0 ]; then
    mapfile -t ready < <(ls "$STAGE"/*.tar 2>/dev/null)          # phase1 done: all complete
  else
    mapfile -t ready < <(ls -t "$STAGE"/*.tar 2>/dev/null | tail -n +2)  # all but newest
  fi
  if [ "${#ready[@]}" -gt 0 ]; then
    printf '%s\n' "${ready[@]}" | xargs -P 16 -I{} bash -c 'extract_one "$@"' _ {}
    echo "[drain] $(date '+%T') +${#ready[@]} extracted; phase1_alive=$alive staged_left=$(ls "$STAGE"/*.tar 2>/dev/null | wc -l) train_files=$(find "$ROOT/train" -type f 2>/dev/null | wc -l)"
  else
    [ "$alive" -eq 0 ] && break
    sleep 5
  fi
done

n=$(find "$ROOT/train" -type f 2>/dev/null | wc -l); c=$(ls "$ROOT/train" 2>/dev/null | wc -l)
echo "[drain] $(date '+%T') DONE train files=$n classes=$c"
rmdir "$STAGE" 2>/dev/null || true
[ "$c" -ge 1000 ] && [ "$n" -ge 1230000 ] && echo "TRAIN_READY" || { echo "[drain] INCOMPLETE (classes=$c files=$n)"; exit 1; }
