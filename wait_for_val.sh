#!/bin/bash
# Gate for the V1-refine run. The ImageNet migration copies TRAIN first, then VAL,
# into data/ImageNet-ILSVRC2012/{train,val} (the GroupTestingDataset symlinks point
# here). NOTE: class-dir symlinks resolve as soon as the target directory exists,
# even while its images are still copying -- so symlink resolution is NOT a
# completeness signal. The real signal is SIZE STABILITY: both trees must reach
# their expected file counts AND stop growing.
#
# Each 30s tick logs file-count/size + delta for BOTH trees so active copying vs.
# a stall is visible. Exits 0 (=> auto-launch the run) only when train and val are
# both near-complete and have held steady for STABLE_TICKS consecutive ticks.
set -u
cd "$(dirname "$0")"
TRAIN=data/ImageNet-ILSVRC2012/train
VAL=data/ImageNet-ILSVRC2012/val
GT=data/GroupTestingDataset

# Full ImageNet-1k targets (underlying tree is the full 1000-class set).
TRAIN_TARGET=1281167
VAL_TARGET=50000
TRAIN_MIN=1230000     # ~96% of full train => tolerate minor class-count variance
VAL_MIN=48000         # ~96% of full val
STABLE_TICKS=3        # consecutive no-growth ticks (both trees) => copy finished

pf_t=0; pb_t=0; pf_v=0; pb_v=0
stable=0

flag() { # $1=delta-bytes $2=files -> growth flag
  if [ "$1" -gt 0 ]; then echo GROWING
  elif [ "$2" -gt 0 ]; then echo STABLE
  else echo empty; fi
}

while true; do
  ts=$(date '+%F %T')
  ft=$(find "$TRAIN" -type f 2>/dev/null | wc -l); bt=$(du -sb "$TRAIN" 2>/dev/null | cut -f1)
  fv=$(find "$VAL"   -type f 2>/dev/null | wc -l); bv=$(du -sb "$VAL"   2>/dev/null | cut -f1)
  dbt=$((bt - pb_t)); dbv=$((bv - pb_v)); dft=$((ft - pf_t)); dfv=$((fv - pf_v))
  gt=$(flag $dbt $ft); gv=$(flag $dbv $fv)
  b=$(( $(find $GT/0/val/ -maxdepth 1 -xtype l 2>/dev/null | wc -l) + $(find $GT/1/val/ -maxdepth 1 -xtype l 2>/dev/null | wc -l) ))
  pctt=$((ft * 100 / TRAIN_TARGET)); pctv=$((fv * 100 / VAL_TARGET))

  echo "$ts  TRAIN ${ft}f (${pctt}%) Δ${dft} [$gt]  |  VAL ${fv}f (${pctv}%) Δ${dfv} [$gv]  |  broken_val_symlinks=$b"

  if [ "$ft" -ge "$TRAIN_MIN" ] && [ "$fv" -ge "$VAL_MIN" ] && [ "$dbt" -eq 0 ] && [ "$dbv" -eq 0 ] && [ "$b" -eq 0 ]; then
    stable=$((stable + 1))
    echo "$ts  both trees complete & steady ($stable/$STABLE_TICKS)"
    if [ "$stable" -ge "$STABLE_TICKS" ]; then
      echo "$ts  DATA_READY  train=${ft}f val=${fv}f"
      exit 0
    fi
  else
    stable=0
  fi

  pf_t=$ft; pb_t=$bt; pf_v=$fv; pb_v=$bv
  sleep 30
done
