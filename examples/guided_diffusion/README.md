# AutoDiffusion for Guided-Diffusion
This is a PyTorch implementation of AutoDiffusion for Guided-Diffusion. The code is heavily based on [Guided-Diffusion codebase](https://github.com/openai/guided-diffusion).

## Checkpoints, Reference Batches and FID Stats
We use checkpoint and reference batches published in the [Guided-Diffusion codebase](https://github.com/openai/guided-diffusion) to search optimal time steps sequence and architecture for guided-diffusion.

| Dataset                        | Checkpoint | Reference Batches | 
| ----------------------------- | ---------------- | -------------------- |
| ImageNet 64x64 | [64x64_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt), [64x64_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt)          | [ImageNet 64x64](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz) |
| LSUN Cat | [lsun_cat.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_cat.pt)         | [LSUN cat](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/cat/VIRTUAL_lsun_cat256.npz) |
| LSUN Bedroom |   [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)      | [LSUN bedroom](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz)  |


Our search process employs FID as a performance estimation. The FID stats used in our experiments can be accessed via links provided in the columns on Google Drive.
| Dataset                        | FID Stats | 
| ----------------------------- | ---------------- | 
| ImageNet 64x64   |[ImageNet 64x64](https://drive.google.com/file/d/1k_YBs7SulpyaaaefQ_dffN9-DpOj3-cl/view?usp=drive_link)|
| LSUN Cat   |[LSUN Cat](https://drive.google.com/file/d/1_mKlQezFR12UrKLLi0uji-iPKM0ChgP5/view?usp=drive_link)|
| LSUN Bedroom |[LSUN bedroom](https://drive.google.com/file/d/1C9seBQ5zq0bVyPXjRXy5U7I_Mz_G1nY9/view?usp=drive_link)|

## [Search for Optimal Time Steps](Search_for_timesteps)
Use the script below to search the optimal time steps. Ensure that you've properly configured `--model_path`, `--classifier_path` (specifically for the class-conditional ImageNet-64 Guided-Diffusion model), and `--ref_path` to point to your downloaded checkpoint and FID stats. You can also set `--time_step` to specify the target length of searched time steps. 

When the search process completes, you can find the file 'log.txt' in the `--save_dir` directory. To identify the best results from each search iteration, search for the keyword 'top' in this file. For optimal outcomes, select the top candidates from the final iteration.

Search time steps sequence for class-conditional ImageNet 64x64 Guided-Diffusion model:
```
bash search_imagenet64_classifier_guidance.sh
```

Search time steps sequence for Guided-Diffusion model on LSUN Cat:
```
bash search_lsun_cat.sh
```

Search time steps sequence for Guided-Diffusion model on LSUN Bedroom:
```
bash search_lsun_bedroom.sh
```

## Search for Time Steps and Architecture
When searching for time steps and architectures, we use the full noise prediction network and search time steps only in the first 5 iterations of the evolutionary search. Then, we search the time steps and model architectures together in the remaining search process. Therefore, ensure that the total `max_epochs` is set to a value greater than 5. You can also set `--time_step` to specify the target length of searched time steps, and `--index_step` to specify the maximum total number of architecture layers (referenced as $N_{max}$ in our paper). 

You can identify the searched results from the 'log.txt' in the `--save_dir` directory as outlined in [previou section](Search_for_timesteps). In this file, 'timesteps' is the searched time steps, while 'skip_layers' is the pruned architecture layers at each time steps. 

Search time steps sequence and architecture for Guided-Diffusion on ImageNet 64x64:
```
bash search_dynamic_unet_imagenet64_classifier_guidance_progressive.sh
```

## Sampling using Searched Results
To sample with your searched time steps, set `--use_timestep` to your searched results in the following script. 

For the class-conditional ImageNet 64x64 Guided-Diffusion model:
```
bash sample_imagenet64_classifier_guidance_subnet.sh
```

For the LSUN cat model:
```
bash sample_LSUN_cat_subnet.sh
```

For the LSUN bedroom model:
```
bash sample_LSUN_bedroom_subnet.sh
```

To sample with both searched time steps and architecture, set `--use_timestep` to your searched time steps and `--skip_layers` to your searched architecture. 

For the class-conditional ImageNet 64x64 Guided-Diffusion model:
```
bash sample_condition_imagenet64_dynamic_subnet.sh
```

## Calculating FID Score for the Generated Samples
Please refer to [Guided-Diffusion codebase](https://github.com/openai/guided-diffusion/tree/main/evaluations).
