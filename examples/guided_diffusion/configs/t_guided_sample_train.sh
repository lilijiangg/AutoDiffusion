#!/usr/bin/env bash
GPUS=$1
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --use_new_attention_order True"
TRAIN_FLAGS="--batch_size 32 --lr 1e-5 --save_interval 2000 --log_interval 100 --weight_decay 0.05 --use_checkpoint False"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $GPUS python3 scripts/image_train.py $MODEL_FLAGS $TRAIN_FLAGS --data_dir "s3://research-model-hh-b/Dataset/ILSVRC2012/imagenet.train.nori.list" --model_dir $2 --save_dir $3 --t_guided True