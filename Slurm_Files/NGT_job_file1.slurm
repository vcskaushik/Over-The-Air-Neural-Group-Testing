#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=NGT_Job1       #Set the job name to "JobExample5"
#SBATCH --time=24:00:00              #Set the wall clock limit to 1hr and 30min

#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH --mem=64G                  #Request 2560MB (2.5GB) per node 128
#SBATCH --output=/scratch/user/vcskaushik9/NGT_Job1Out.%j      #Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:a100:2       #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=132776798961      #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=vcskaushik9@tamu.edu    #Send all emails to email_address 

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


echo "Starting Training Group 4"

python -u main4.py --background-K 3 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 32 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXt_K3_A4_2 --dist-url 'tcp://127.0.0.1:7101' --dist-backend 'nccl' --multiprocessing-distributed --epochs 200 --world-size 1 --rank 0 > Log_files/log_ResNeXt_K3_A4_2.txt 2>&1 || true


echo "Completed Training Group 4"

echo "Starting Training Group 4 SNR 0"

python -u main4.py --background-K 3 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 32 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXt_K3_A4_SNR0_2 --dist-url 'tcp://127.0.0.1:7101' --dist-backend 'nccl' --multiprocessing-distributed --epochs 200 --SNR 0 --world-size 1 --rank 0  > Log_files/log_ResNeXt_K3_A4_SNR0_2.txt 2>&1 || true


echo "Completed Training Group 4 SNR 0"
