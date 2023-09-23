#!/usr/bin/env bash
GPUS=$1
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 64 --num_heads 2 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 32 --microbatch 16 --lr 3e-4 --save_interval 10000 --log_interval 100 --weight_decay 0.05 --use_checkpoint True --mix_loss True"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $GPUS python3 scripts/super_res_train.py $MODEL_FLAGS $TRAIN_FLAGS --data_dir /data/jupyter/data/DIV2K/DIV2K_train_HR_sub/ --model_path $2 --save_dir $3