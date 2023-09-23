MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --in_channels 64 --use_feature True --image_size 256 --learn_sigma False --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 32 --num_samples 64 --in_steps -1 --timestep_respacing 250 --use_ddim False --res True"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $1 python3 scripts/fdpm_image_sample.py $MODEL_FLAGS --model_path $2 $SAMPLE_FLAGS --save_dir $3