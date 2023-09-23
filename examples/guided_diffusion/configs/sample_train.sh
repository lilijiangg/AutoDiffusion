#!/usr/bin/env bash 
GPUS=$1
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 4 --lr 5e-5 --save_interval 10000 --log_interval 100 --weight_decay 0.05 --use_checkpoint True --res True"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $GPUS python3 scripts/image_train.py $MODEL_FLAGS $TRAIN_FLAGS --data_dir "s3://research-model-hh-b/Dataset/ILSVRC2012/imagenet.train.nori.list" --save_dir $2