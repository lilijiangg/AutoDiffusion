# AutoDiffusion for Stable Diffusion
This is a PyTorch implementation of AutoDiffusion for Stable Diffusion. The code is heavily based on [Stable Diffusion codebase](https://github.com/CompVis/stable-diffusion).

## Requirements
To set up the required environment:
```
conda env create -f environment.yml
conda activate ldm
pip install -e .
```

## Checkpoints, datasets and FID Stats
We use the `sd-v1-4.ckpt` checkpoint from the [Stable Diffusion codebase](https://github.com/CompVis/stable-diffusion) for searching the optimal time steps sequence for stable diffusion.

We utilize the validation set of COCO dataset to search optimal time steps and estimate the FID score of searched results for stable diffusion. Please follow this [guide](https://github.com/kakaobrain/rq-vae-transformer/blob/main/data/README.md#ms-coco) to download the dataset. Once downloaded, the data structure should appear as:
```
path/to/dataset
├── captions_val2014_30K_samples.json
├── val2014
│  ├── COCO_val2014_xx.jpg
│  ├── ...
```

Our search process employs FID as a performance estimation. The FID stats of COCO datasets used in our experiments can be accessed via this [link](https://drive.google.com/drive/folders/1tza3EPmxYTLVNErKRrgWfmVu0XbNynMm) on Google Drive.

## Search for Optimal Time Steps
Before starting the search process, ensure that `data_root` in `configs/stable-diffusion/v1-inference_coco.yaml` is correctly configured to point to the path of your downloaded dataset. Then, follow the scripts below to search for the optimal time steps. It's essential to set both `n_samples` in these scripts and `batch_size` in `configs/stable-diffusion/v1-inference_coco.yaml` to desired your batch size.

Search time steps sequence for Stable Diffusion on COCO dataset with DPM-Solver:
```
bash search_dpm_solver.sh
```

Search time steps sequence for Stable Diffusion on COCO dataset with PLMS:
```
bash search_plms.sh
```

Parameters:
- `ckpt`: Path to the checkpoint of Stable Diffusion.
- `outdir`: Path to the searched results.
- `n_samples`: The batch size. 
- `num_sample`: Number of samples used for estimating FID score in the search process. 
- `time_step`: Target length of the searched time steps sequence. 
- `max_epochs`: Maximum number of iterations for the search process.
- `ref_mu` and `ref_sigma`: Paths to the downloaded FID stats.

When the search process completes, you can find the file `log.txt` in the `--outdir` directory. To identify the best results from each search iteration, search for the keyword `top` in this file. For the most optimal results, select the top candidates from the final iteration.


## Sampling using Searched Results
For Stable Diffusion on COCO dataset with DPM-Solver:
```
bash sample_fid_dpm_solver.sh
```

For Stable Diffusion on COCO dataset with PLMS:
```
bash sample_fid_plms.sh
```

Parameters:
- `ckpt`: Path to the checkpoint of Stable Diffusion.
- `outdir`: Path to the generated samples.
- `n_samples`: The batch size. 
- `num_sample`: Total number of samples to be generated.
- `ddim_steps`: The number of time steps. 
- `timesteps`: Set this parameter to `"[xxx]"` where `[xxx]` is the searched results. When this parameter is set to `""`, the samples will be generated using uniform time steps.

## Calculating FID Score for the Generated Samples
For calculating the FID score of generated samples, please refer to [pytorch-fid](https://github.com/mseitzer/pytorch-fid).
