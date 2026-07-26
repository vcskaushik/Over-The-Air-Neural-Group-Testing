#!/bin/bash
# Handoff step 3 for BEST=30: 2-epoch utility refresh of lam30 (+lam0 control) ->
# Kaiming Stage C on refreshed, + worst-case (adv pretrained) Stage C on unrefreshed
# lam30; THEN the key number: worst-case (adv pretrained) Stage C on the REFRESHED
# lam30 checkpoint (refresh-durability under the strong attacker).
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
COMMON="--data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 -a resnet18"
export CUDA_VISIBLE_DEVICES=0
BEST=30

echo "########## run_refresh_and_worstcase.sh $BEST ##########"
bash run_refresh_and_worstcase.sh $BEST
echo "=== refresh_and_worstcase rc=$? ==="

echo "########## pretrained-adv Stage C on REFRESHED lam${BEST} ##########"
SCW=Trained_Models/StageC_RefreshedInvariance_lam${BEST}_advpretrained
$PY -u -m privacy.eval_privacy \
    --stage-b-ckpt Trained_Models/RefreshedInvariance_lam${BEST}/checkpoint.pth.tar $COMMON \
    --stage-c-epochs 30 --batch-size 32 --adv-init pretrained -j 8 -valj 4 \
    --output_dir "$SCW" > ${SCW}_stdout.log 2>&1
echo "=== refreshed-BEST pretrained-adv rc=$? ==="

echo "REFRESH_BEST_DONE $BEST" > Trained_Models/.refresh_best_done
echo "===== REFRESH BEST ($BEST) COMPLETE ====="
