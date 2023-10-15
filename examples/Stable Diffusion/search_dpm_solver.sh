CUDA_VISIBLE_DEVICES=0 \
python scripts/search_ea.py --dpm_solver \
--ckpt 'path/to/model.ckpt' \
--outdir 'outputs_search/search_step10_numsaple1000_dpm_solver' \
--config './configs/stable-diffusion/v1-inference_coco.yaml' \
--n_samples 6 \
--num_sample 1000 \
--time_step 10 \
--max_epochs 10 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 \
--seed 0 \
--use_ddim_init_x True \
--ref_mu 'path/to/coco2014_mu.npy' \
--ref_sigma 'path/to/coco2014_sigma.npy' \