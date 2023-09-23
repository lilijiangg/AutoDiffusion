#!/usr/bin/env bash

GPUS=$1

MODEL_FLAGS_512_upsample="--attention_resolutions 32,16 --class_cond True --diffusion_steps 1000 --large_size 512 --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 3 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --rescale_learned_sigmas False --rescale_timesteps False"

MODEL_FLAGS_256_upsample="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --rescale_learned_sigmas False --rescale_timesteps False"

MODEL_FLAGS_256_cond="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --rescale_learned_sigmas False --rescale_timesteps False"

MODEL_FLAGS_256_uncond="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --rescale_learned_sigmas False --rescale_timesteps False"

MODEL_FLAGS_512_cond="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 512 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --rescale_learned_sigmas False --rescale_timesteps False"

# upsample flops
#python3 calc_flops.py $MODEL_FLAGS_256_upsample --base_samples ./

# image generation flops
#python3 calc_flops.py $MODEL_FLAGS_256_uncond

SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing ddim25 --use_ddim True"
PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $GPUS python3 --base_samples /data/jupyter/data/DIV2K/DIV2K_valid_LR_bicubic/X4_sub/