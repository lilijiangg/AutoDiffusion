MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 100 --num_samples 1000 --use_ddim True" 

CUDA_VISIBLE_DEVICES=0 \
python search_dynamic_unet_imagenet64_classifier_guidance_progressive.py $MODEL_FLAGS --classifier_scale 1.0 \
--classifier_path path/to/64x64_classifier.pt --classifier_depth 4 \
--model_path path/to/64x64_diffusion.pt $SAMPLE_FLAGS \
--ref_path path/to/imagenet_ref_stats.pkl \
--save_dir './exps/imagenet64_cond/ea_search/N_max580' \
--time_step 10 \
--max_epochs 15 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 \
--seed 0 \
--m_prob 0.25 \
--use_ddim_init_x True \
--use_dynamic_unet True \
--index_step 580 \
--max_prun=0.1 \
--min_prun=0.0 \
--MASTER_PORT '12345' \