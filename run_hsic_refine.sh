#!/bin/bash
# V1 REFINEMENT (ResNet-18, ITIT, sigma=0).
# Two gaps in the first sweep (see docs/superpowers/results/2026-07-25-*.md):
#   1. lambda_H grid skipped the entire 10->100 knee -> "no free-lunch" unproven.
#   2. worst-case (adv-init pretrained) leakage was measured for only 3 checkpoints,
#      while the headline led with the weak Kaiming attacker.
# This script fills both: a fine lambda_H grid {20,30,50,70}, and adv-init
# pretrained Stage C across the WHOLE sweep. Idempotent (skip-if-exists), so it
# reuses restored coarse-lambda checkpoints and only trains the new points.
#
# Prereqs (see HANDOFF-v1-refine.md): .venv built, dataset rebuilt, and the
# coarse checkpoints restored into Trained_Models/ from the Drive backup.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
DATA=data/GroupTestingDataset
COMMON="--data $DATA --task-num 2 --background-K 0 --GT-alg 1 -a resnet18"
STAGEA=Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar
export CUDA_VISIBLE_DEVICES=0

FINE="20 30 50 70"
ALL="0 1 10 20 30 50 70 100 1000"

# ---- Fine grid: train -> utility(48k) -> Kaiming Stage C (mirrors run_hsic_sweep.sh) ----
for LAM in $FINE; do
  OUT=Trained_Models/Invariance_ResNet18_lam${LAM}
  CKPT=$OUT/invariance_final.pth.tar
  echo "########## fine lambda_H=$LAM ##########"
  if [ ! -f "$CKPT" ]; then
    $PY -u -m privacy.train_invariance --stage-a-ckpt $STAGEA $COMMON \
        --hsic-lambda $LAM --hsic-extractor pretrained --hsic-num-random 2 \
        --pk-classes 8 --pk-per-class 4 --pk-firearm 4 \
        --epochs 30 --enc-lr 1e-4 --rec-lr 1e-4 --batch-size 32 -j 8 -valj 4 \
        --output_dir $OUT --log-name invariance.log > ${OUT}_train_stdout.log 2>&1
    echo "train exit=$? lam=$LAM"
  else
    echo "ckpt exists, skip train lam=$LAM"
  fi
  $PY -u main.py --evaluate --resume $CKPT $COMMON --batch-size 32 -valj 4 \
      > ${OUT}_utility.log 2>&1
  echo "utility exit=$? lam=$LAM"
  SC=Trained_Models/StageC_Invariance_ResNet18_lam${LAM}
  if [ ! -f "$SC/leakage.json" ]; then
    $PY -u -m privacy.eval_privacy --stage-b-ckpt $CKPT $COMMON \
        --stage-c-epochs 30 --batch-size 32 --adv-init kaiming -j 8 -valj 4 \
        --output_dir $SC > ${SC}_stdout.log 2>&1
    echo "stageC-kaiming exit=$? lam=$LAM"
  fi
done

# ---- Worst-case: adv-init pretrained Stage C across the WHOLE sweep ----
SCB=Trained_Models/StageC_OnStageA_ResNet18_advpretrained
if [ ! -f "$SCB/leakage.json" ]; then
  echo "########## worst-case baseline (Stage A) ##########"
  $PY -u -m privacy.eval_privacy --stage-b-ckpt $STAGEA $COMMON \
      --stage-c-epochs 30 --batch-size 32 --adv-init pretrained -j 8 -valj 4 \
      --output_dir $SCB > ${SCB}_stdout.log 2>&1
  echo "worstcase-baseline exit=$?"
fi
for LAM in $ALL; do
  CKPT=Trained_Models/Invariance_ResNet18_lam${LAM}/invariance_final.pth.tar
  SCW=Trained_Models/StageC_Invariance_ResNet18_lam${LAM}_advpretrained
  if [ ! -f "$CKPT" ]; then echo "MISSING ckpt lam=$LAM — restore from Drive"; continue; fi
  if [ -f "$SCW/leakage.json" ]; then echo "worstcase lam=$LAM exists, skip"; continue; fi
  echo "########## worst-case lambda_H=$LAM ##########"
  $PY -u -m privacy.eval_privacy --stage-b-ckpt $CKPT $COMMON \
      --stage-c-epochs 30 --batch-size 32 --adv-init pretrained -j 8 -valj 4 \
      --output_dir $SCW > ${SCW}_stdout.log 2>&1
  echo "worstcase-stageC exit=$? lam=$LAM"
done

echo "REFINE_DONE" > Trained_Models/.refine_done
echo "===== REFINE COMPLETE ====="
echo "Next: pick BEST fine-grid lambda (largest worst-case leakage drop at recall >= ~48/50),"
echo "then: bash run_refresh_and_worstcase.sh <BEST>   # refresh + refresh-worst-case"
