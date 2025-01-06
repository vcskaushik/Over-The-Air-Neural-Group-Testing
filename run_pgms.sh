
#CUDA_VISIBLE_DEVICES=0,1 python main.py --background-K 0 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 32 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXt101FullK0_mean --dist-url 'tcp://127.0.0.1:7101' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 > Log_files/log_K0_mean.txt 2>&1

#CUDA_VISIBLE_DEVICES=0,1 python main.py --background-K 1 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 32 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXt101FullK2_A2_SNR5 --dist-url 'tcp://127.0.0.1:7101' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 > Log_files/log_K2_A2_SNR5.txt 2>&1

#CUDA_VISIBLE_DEVICES=1 python main.py --background-K 0 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 15 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Val_Files/VAL_DEBUG_P0.01pct_K0_mean --evaluate --resume Trained_Models/ResNeXt101FullK0_mean/checkpoint.pth.tar

#CUDA_VISIBLE_DEVICES=1 python main.py --background-K 1 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 15 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Val_Files/VAL_DEBUG_P0.01pct_K1A1D2_SNR5 --evaluate --resume Trained_Models/ResNeXt101FullK2_A2_SNR5/checkpoint.pth.tar

#python results_plot.py

echo "Starting Training Group 16, SNR 5"

CUDA_VISIBLE_DEVICES=0,1 python main.py --background-K 15 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 4 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXtFullK15_A2_SNR5 --dist-url 'tcp://127.0.0.1:7101' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --epochs 300 --SNR 5 > Log_files/log_ResNeXt2_K15_A2_SNR5.txt 2>&1 || true

echo "Completed Training Group 16, SNR 5"

echo "Starting Training Group 8, SNR 5"

CUDA_VISIBLE_DEVICES=0,1 python main.py --background-K 7 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 8 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXtFullK7_A2_SNR5 --dist-url 'tcp://127.0.0.1:7101' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --epochs 300 --SNR 5 > Log_files/log_ResNeXt2_K7_A2_SNR5.txt 2>&1 || true

echo "Completed Training Group 8, SNR 5"

echo "Starting Training Group 4, SNR 5"

CUDA_VISIBLE_DEVICES=0,1 python main.py --background-K 3 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 16 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXtFullK3_A2_SNR5 --dist-url 'tcp://127.0.0.1:7101' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --SNR 5 > Log_files/log_ResNeXt2_K3_A2_SNR5.txt 2>&1 || true

echo "Completed Training Group 4, SNR 5"



