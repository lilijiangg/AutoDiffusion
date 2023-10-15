CUDA_VISIBLE_DEVICES=0 \
python scripts/search_ea.py --plms \
--ckpt 'path/to/model.ckpt' \
--outdir 'outputs_search/search_step4_numsaple2000_plms' \
--config './configs/stable-diffusion/v1-inference_coco.yaml' \
--n_samples 6 \
--num_sample 2000 \
--time_step 4 \
--max_epochs 10 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 15 \
--seed 0 \
--use_ddim_init_x True \
--ref_mu 'path/to/coco2014_mu.npy' \
--ref_sigma 'path/to/coco2014_sigma.npy' \