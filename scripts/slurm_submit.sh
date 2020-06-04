#!/bin/sh

#SBATCH --partition=gpu-medium
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=none
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=6

export HOME_DIR="/home/s1738291/nlpclassics"
export DATA_DIR="/data/s1738291/nlpclassics"

cd $HOME_DIR
module load PyTorch/1.3.1-fosscuda-2019b-Python-3.7.4

pip install --user pytorch_lightning==0.7.6 \
    wandb==0.8.35 \
    pyconll==2.2.1

export WANDB_API_KEY="a71e7c1fa2e0e56225284ec622b82ebe5a079323"

python train.py --gpus 1 --max_epochs 3 --num_nodes 1 --val_check_interval 0.25 --track --data-dir $DATA_DIR