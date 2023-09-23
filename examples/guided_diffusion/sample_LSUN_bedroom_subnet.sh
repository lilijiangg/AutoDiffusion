MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True" 
SAMPLE_FLAGS="--batch_size 36 --num_samples 50000 --use_ddim True"

CUDA_VISIBLE_DEVICES=0 \
python scripts/image_sample.py $MODEL_FLAGS \
--model_path 'path/to/lsun_bedroom.pt' $SAMPLE_FLAGS \
--save_dir './exps/LSUN_bedroom/ea_search/timestep15_m_prob0.25_numsample3000/samples_50000' \
--use_timestep '[644, 737, 67, 804, 134, 871, 6, 639, 268, 335, 402, 469, 536, 603, 670]' \
--port '12348'
