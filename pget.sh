#!/bin/bash
# Parallel chunked HTTP downloader with per-chunk resume, for slow origins that
# cap per-connection bandwidth (image-net.org ~3MB/s/conn). Splits into fixed-size
# byte-range chunks fetched by PAR parallel curls, then concatenates in order.
# Usage: pget.sh URL OUT [PAR] [CHUNK_MB]
set -u
URL=$1; OUT=$2; PAR=${3:-16}; CHUNKMB=${4:-400}
CHUNK=$((CHUNKMB * 1024 * 1024))
SIZE=$(curl -sI "$URL" | awk -F': ' 'tolower($1)=="content-length"{print $2}' | tr -d '\r')
case "$SIZE" in ''|*[!0-9]*) echo "[pget] no content-length for $URL"; exit 1;; esac
PD="${OUT}.parts"; mkdir -p "$PD"
N=$(( (SIZE + CHUNK - 1) / CHUNK ))
echo "[pget] $(date '+%T') $OUT size=$SIZE ($(numfmt --to=iec $SIZE)) chunks=$N par=$PAR"

export URL SIZE CHUNK PD
seq 0 $((N-1)) | xargs -P "$PAR" -I{} bash -c '
  i={}; start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 ))
  [ $end -ge $SIZE ] && end=$(( SIZE - 1 ))
  exp=$(( end - start + 1 )); p="$PD/part_$(printf %05d $i)"
  if [ -f "$p" ] && [ "$(stat -c%s "$p" 2>/dev/null)" = "$exp" ]; then exit 0; fi
  for a in 1 2 3 4 5 6; do
    curl -s -r ${start}-${end} -o "$p" "$URL"
    [ "$(stat -c%s "$p" 2>/dev/null || echo 0)" = "$exp" ] && exit 0
    sleep 5
  done
  echo "[pget] chunk $i FAILED (want $exp got $(stat -c%s "$p" 2>/dev/null || echo 0))" >&2
  exit 1
' || { echo "[pget] $(date '+%T') one or more chunks failed"; exit 1; }

echo "[pget] $(date '+%T') assembling $N chunks -> $OUT"
cat "$PD"/part_* > "$OUT"
got=$(stat -c%s "$OUT")
if [ "$got" = "$SIZE" ]; then
  rm -rf "$PD"; echo "[pget] $(date '+%T') DONE $OUT $got bytes"
else
  echo "[pget] SIZE MISMATCH got=$got want=$SIZE (parts kept for resume)"; exit 1
fi
