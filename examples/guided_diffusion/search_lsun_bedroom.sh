MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True" 
SAMPLE_FLAGS="--batch_size 36 --num_samples 3000 --use_ddim True"

CUDA_VISIBLE_DEVICES=0 \
python search_uncondition_model.py $MODEL_FLAGS \
--model_path 'path/to/lsun_bedroom.pt' $SAMPLE_FLAGS \
--save_dir './exps/LSUN_bedroom/ea_search_ddim/timestep15_m_prob0.25_numsample3000' \
--time_step 5 \
--max_epochs 10 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 \
--seed 0 \
--m_prob 0.25 \
--ref_path 'path/to/lsun_bedroom_ref_stats.pkl' \
--use_ddim_init_x True \
--MASTER_PORT '12345' \