#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=NGT_Job2_2       #Set the job name to "JobExample5"
#SBATCH --time=48:00:00              #Set the wall clock limit to 1hr and 30min

#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH --mem=64G                  #Request 2560MB (2.5GB) per node 128
#SBATCH --output=/scratch/user/vcskaushik9/NeuralGroupTesting/Slurm_Files/Out_Files/NGT_Job1Out.%j      #Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:a100:2       #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132776798961      #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=vcskaushik9@tamu.edu    #Send all emails to email_address 

#First Executable Line
cd $SCRATCH

module load CUDA/11.7.0
module load Anaconda3/2022.05

source activate ngt_env3

module load WebProxy

export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

cd NeuralGroupTesting

nvidia-smi 

echo "Starting Training Group 1 SNR 1"

CUDA_VISIBLE_DEVICES=0,1 python -u main_dist.py --background-K 7 --GT-alg 4 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 32 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXt_K7_A4_SNR1_2 --dist-url 'tcp://127.0.0.1:7105' --dist-backend 'nccl' --multiprocessing-distributed --epochs 200 --SNR 1 --world-size 1 --rank 0  > Log_files/log_ResNeXt_K7_A4_SNR1_2.txt 2>&1 || true


echo "Completed Training Group 1 SNR 1"

echo "Starting Training Group 1 SNR 2"

CUDA_VISIBLE_DEVICES=0,1 python -u main_dist.py --background-K 7 --GT-alg 4 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 32 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXt_K7_A4_SNR2_2 --dist-url 'tcp://127.0.0.1:7105' --dist-backend 'nccl' --multiprocessing-distributed --epochs 200 --SNR 2 --world-size 1 --rank 0  > Log_files/log_ResNeXt_K7_A4_SNR2_2.txt 2>&1 || true

echo "Completed Training Group 1 SNR 2"
