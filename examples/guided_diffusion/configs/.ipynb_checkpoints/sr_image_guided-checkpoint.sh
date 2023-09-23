MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --timestep_respacing 250 --use_ddim False --scale 4 --use_checkpoint True --prompt_flag None --lq_embed True"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
torchrun --nproc_per_node=4 scripts/sr_sample_image_guided.py $MODEL_FLAGS --model_path $1 $SAMPLE_FLAGS --base_samples $2 --save_dir $3 --reverse $4 --guided_scale $5