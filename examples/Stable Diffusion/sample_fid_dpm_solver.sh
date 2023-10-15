CUDA_VISIBLE_DEVICES=0 \
python scripts/txt2img_fid.py --dpm_solver \
--ckpt 'path/to/model.ckpt' \
--outdir 'outputs/search_step4_numsaple1000_dpm_solver_searched_results' \
--config './configs/stable-diffusion/v1-inference_coco.yaml' \
--n_samples 6 \
--num_sample 10000 \
--cal_fid True \
--ddim_steps 4 \
--timesteps '[0.014986000955104828, 0.2857150137424469, 0.5005000233650208, 0.7182819843292236, 0.9260739684104919]' \
# --timesteps '' \