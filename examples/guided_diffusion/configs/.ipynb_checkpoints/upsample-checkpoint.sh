MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 4192 --timestep_respacing 250 --use_ddim False --scale 4"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $1 python3 scripts/super_res_sample.py $MODEL_FLAGS --model_path $2 $SAMPLE_FLAGS --base_samples /data/jupyter/data/DIV2K/DIV2K_valid_LR_bicubic/X4_sub/ --save_dir $3 --reverse True