#!/bin/bash
#SBATCH --mem=64G
. /home/urso/PavicHDR/venv/bin/activate
python /home/urso/PavicHDR/test.py \
--dataset_dir /home/urso/Datasets/Kalantari \
--ckpt /home/urso/PavicHDR/runs/train_2/best_checkpoint.pth \