MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 128 --num_samples 50000 --use_ddim True"

CUDA_VISIBLE_DEVICES=0 \
python ./scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 \
--classifier_path path/to/imagenet64_cond/64x64_classifier.pt --classifier_depth 4 \
--model_path path/to/64x64_diffusion.pt $SAMPLE_FLAGS \
--use_mean False \
--MASTER_PORT '12347' \
--save_dir './exps/imagenet64_cond/ea_search_random_ddim/timestep6_numsample5000_p500/sample_50000' \
--use_timestep '[94, 834, 217, 944, 574, 354]' \


