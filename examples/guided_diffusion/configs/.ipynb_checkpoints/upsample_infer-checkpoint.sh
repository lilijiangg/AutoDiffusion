MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --rescale_learned_sigmas False --rescale_timesteps False"
SAMPLE_FLAGS="--batch_size 8 --num_samples 1000 --timestep_respacing ddim25 --use_ddim True --save_dir div2k_valid_ddim25"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $1 python3 scripts/super_res_infer.py $MODEL_FLAGS --model_path /data/jupyter/models/64_256_upsampler.pt $SAMPLE_FLAGS --base_samples /data/jupyter/data/DIV2K/DIV2K_valid_LR_bicubic/X4_sub/ --scale 4