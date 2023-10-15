CUDA_VISIBLE_DEVICES=0 \
python scripts/txt2img_fid.py --plms \
--ckpt 'path/to/model.ckpt' \
--outdir 'outputs/uniform_step10_samples10000_uniform' \
--config './configs/stable-diffusion/v1-inference_coco.yaml' \
--n_samples 6 \
--num_sample 10000 \
--cal_fid True \
--ddim_steps 10 \
--timesteps '' \