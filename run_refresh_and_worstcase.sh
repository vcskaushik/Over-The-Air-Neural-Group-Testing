#!/bin/bash
# §3f refresh-fragility diagnostic + §3e worst-case (adv-init pretrained).
# Usage: bash run_refresh_and_worstcase.sh <BEST_LAM>
# For BEST_LAM and the lam=0 control: 2-epoch utility refresh -> Stage C.
# Plus one adv-init pretrained Stage C on BEST_LAM (strongest-privacy) checkpoint.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
DATA=data/GroupTestingDataset
COMMON="--data $DATA --task-num 2 --background-K 0 --GT-alg 1 -a resnet18"
export CUDA_VISIBLE_DEVICES=0
BEST=${1:-100}

for LAM in $BEST 0; do
  CKPT=Trained_Models/Invariance_ResNet18_lam${LAM}/invariance_final.pth.tar
  ROUT=Trained_Models/RefreshedInvariance_lam${LAM}
  echo "###### refresh lam=$LAM ######"
  # 2-epoch utility-only refresh (E+R unfrozen, firearm CE)
  $PY -u main.py --resume $CKPT $COMMON \
      --epochs 2 --batch-size 32 --lr 0.0001 -j 8 -valj 4 \
      --output_dir $ROUT --log-name refresh.log > ${ROUT}_stdout.log 2>&1
  echo "refresh exit=$? lam=$LAM"
  # utility of refreshed
  $PY -u main.py --evaluate --resume $ROUT/checkpoint.pth.tar $COMMON --batch-size 32 -valj 4 \
      > ${ROUT}_utility.log 2>&1
  echo "refresh-utility exit=$? lam=$LAM"
  # Stage C on refreshed
  SC=Trained_Models/StageC_RefreshedInvariance_lam${LAM}
  $PY -u -m privacy.eval_privacy --stage-b-ckpt $ROUT/checkpoint.pth.tar $COMMON \
      --stage-c-epochs 30 --batch-size 32 --adv-init kaiming -j 8 -valj 4 \
      --output_dir $SC > ${SC}_stdout.log 2>&1
  echo "refresh-stageC exit=$? lam=$LAM"
done

# §3e worst-case: adv-init pretrained Stage C on strongest-privacy (BEST) checkpoint
echo "###### worst-case adv-init pretrained on lam=$BEST ######"
SCW=Trained_Models/StageC_Invariance_ResNet18_lam${BEST}_advpretrained
$PY -u -m privacy.eval_privacy \
    --stage-b-ckpt Trained_Models/Invariance_ResNet18_lam${BEST}/invariance_final.pth.tar $COMMON \
    --stage-c-epochs 30 --batch-size 32 --adv-init pretrained -j 8 -valj 4 \
    --output_dir $SCW > ${SCW}_stdout.log 2>&1
echo "worstcase exit=$? lam=$BEST"

echo "REFRESH_ALL_DONE" > Trained_Models/.refresh_done
echo "===== REFRESH+WORSTCASE COMPLETE ====="
