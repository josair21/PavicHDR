#!/bin/bash
#SBATCH --mem=64G
. /home/urso/PavicHDR/venv/bin/activate
python /home/urso/PavicHDR/train.py \
--dataset_dir /home/urso/Datasets/Kalantari/ \
--sub_set sig17_training_crop128_stride64 \
--batch_size 32 \
--logdir /home/urso/PavicHDR/runs/train_3
