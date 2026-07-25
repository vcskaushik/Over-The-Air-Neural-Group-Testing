#!/bin/bash
# HSIC invariance v1 sweep orchestrator (ResNet-18, ITIT, sigma=0).
# For each lambda_H: (3c) train invariance -> (3d) utility eval -> (3e) Stage C leakage.
# Faithful to HANDOFF.md commands. Logs per-lambda; writes done-markers.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
DATA=data/GroupTestingDataset
COMMON="--data $DATA --task-num 2 --background-K 0 --GT-alg 1 -a resnet18"
STAGEA=Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar
export CUDA_VISIBLE_DEVICES=0

for LAM in 0 1 10 100 1000; do
  OUT=Trained_Models/Invariance_ResNet18_lam${LAM}
  CKPT=$OUT/invariance_final.pth.tar
  echo "############ lambda_H=$LAM ############"

  # --- 3c: HSIC invariance training ---
  if [ ! -f "$CKPT" ]; then
    $PY -u -m privacy.train_invariance \
        --stage-a-ckpt $STAGEA $COMMON \
        --hsic-lambda $LAM --hsic-extractor pretrained --hsic-num-random 2 \
        --pk-classes 8 --pk-per-class 4 --pk-firearm 4 \
        --epochs 30 --enc-lr 1e-4 --rec-lr 1e-4 --batch-size 32 -j 8 -valj 4 \
        --output_dir $OUT --log-name invariance.log \
        > ${OUT}_train_stdout.log 2>&1
    echo "train exit=$? for lam=$LAM"
  else
    echo "ckpt exists, skip train lam=$LAM"
  fi

  # --- 3d: utility eval (48k protocol) ---
  $PY -u main.py --evaluate --resume $CKPT $COMMON --batch-size 32 -valj 4 \
      > Trained_Models/Invariance_ResNet18_lam${LAM}_utility.log 2>&1
  echo "utility exit=$? for lam=$LAM"

  # --- 3e: Stage C honest leakage ---
  SC=Trained_Models/StageC_Invariance_ResNet18_lam${LAM}
  $PY -u -m privacy.eval_privacy \
      --stage-b-ckpt $CKPT $COMMON \
      --stage-c-epochs 30 --batch-size 32 --adv-init kaiming -j 8 -valj 4 \
      --output_dir $SC > ${SC}_stdout.log 2>&1
  echo "stageC exit=$? for lam=$LAM"

  echo "lam=$LAM DONE" >> Trained_Models/.sweep_progress
done

echo "ALL_DONE" > Trained_Models/.sweep_done
echo "===== SWEEP COMPLETE ====="
