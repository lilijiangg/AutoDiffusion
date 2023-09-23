MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 64 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python3 -m torch.distributed.launch --nproc_per_node=$1 scripts/image_sample.py $MODEL_FLAGS --model_path $2 $SAMPLE_FLAGS --save_dir $3