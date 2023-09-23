MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 100 --num_samples 5000 --use_ddim True"

CUDA_VISIBLE_DEVICES=0 \
python search_imagenet64_classifier_guidance.py $MODEL_FLAGS \
--classifier_path path/to/64x64_classifier.pt --classifier_depth 4 \
--model_path path/to/64x64_diffusion.pt $SAMPLE_FLAGS \
--ref_path path/to/imagenet_ref_stats.pkl \
--save_dir './exps/imagenet64_cond/ea_search/timestep4_numsample5000_prob0.25_cfg1.0' \
--time_step 4 \
--max_epochs 10 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 15 \
--seed 0 \
--m_prob 0.25 \
--use_ddim_init_x True \
--thres 0.2 \
--classifier_scale 1.0 \
--MASTER_PORT '12345' \