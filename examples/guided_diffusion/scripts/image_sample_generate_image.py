"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    diffusion_defaults,
    create_gaussian_diffusion,
)

import time

def reset_diffusion(use_timesteps, active_diffusion, base_diffusion):
    use_timesteps = set(use_timesteps)
    active_diffusion.timestep_map = []
    last_alpha_cumprod = 1.0
    new_betas = []

    active_diffusion.use_timesteps = set(use_timesteps)

    for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod) # 通过长序列的 \overline{alpha} 解 短序列的 \beta
            last_alpha_cumprod = alpha_cumprod
            active_diffusion.timestep_map.append(i)

    import numpy as np
    new_betas = np.array(new_betas, dtype=np.float64)

    active_diffusion.betas = new_betas
    assert len(new_betas.shape) == 1, "betas must be 1-D"
    assert (new_betas > 0).all() and (new_betas <= 1).all()

    active_diffusion.num_timesteps = int(new_betas.shape[0])

    alphas = 1.0 - new_betas  # alpha 递减
    active_diffusion.alphas_cumprod = np.cumprod(alphas, axis=0) # overliane_{x}
    active_diffusion.alphas_cumprod_prev = np.append(1.0, active_diffusion.alphas_cumprod[:-1]) # alpha[0], alpha[0], alpha[1], ...., alpha[T-1]
    active_diffusion.alphas_cumprod_next = np.append(active_diffusion.alphas_cumprod[1:], 0.0)  # alpha[1], alpha[2], ..., alpha[T], alpha[T]
    assert active_diffusion.alphas_cumprod_prev.shape == (active_diffusion.num_timesteps,)  

    # calculations for diffusion q(x_t | x_{t-1}) and others
    active_diffusion.sqrt_alphas_cumprod = np.sqrt(active_diffusion.alphas_cumprod)  # \sqrt{\overline{\alpha}}
    active_diffusion.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - active_diffusion.alphas_cumprod) # \sqrt{1 - \overline{\alpha}}
    active_diffusion.log_one_minus_alphas_cumprod = np.log(1.0 - active_diffusion.alphas_cumprod)  # \log{1 - \overline{\alpha}}
    active_diffusion.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / active_diffusion.alphas_cumprod)      # \frac{1}{\sqrt{\overline{\alpha}}}
    active_diffusion.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / active_diffusion.alphas_cumprod - 1)   

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    active_diffusion.posterior_variance = (
        new_betas * (1.0 - active_diffusion.alphas_cumprod_prev) / (1.0 - active_diffusion.alphas_cumprod)  # DDPM 式7 的 \hat{\beta}
    )
    # log calculation clipped because the posterior variance is 0 at the
    # beginning of the diffusion chain.
    if len(active_diffusion.posterior_variance) > 1:
        active_diffusion.posterior_log_variance_clipped = np.log(
            np.append(active_diffusion.posterior_variance[1], active_diffusion.posterior_variance[1:])
        )
    else:
        active_diffusion.posterior_log_variance_clipped = active_diffusion.posterior_variance
    active_diffusion.posterior_mean_coef1 = (
        new_betas * np.sqrt(active_diffusion.alphas_cumprod_prev) / (1.0 - active_diffusion.alphas_cumprod)
    )
    active_diffusion.posterior_mean_coef2 = (
        (1.0 - active_diffusion.alphas_cumprod_prev)
        * np.sqrt(alphas)
        / (1.0 - active_diffusion.alphas_cumprod)
    )

def main():
    args = create_argparser().parse_args()

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    dist_util.setup_dist()
    logger.configure(args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.use_timestep is not None:
        args.use_timestep = eval(args.use_timestep)
        args.use_timestep = sorted(args.use_timestep)
        import copy
        base_diffusion = copy.deepcopy(diffusion)
        reset_diffusion(use_timesteps=args.use_timestep, active_diffusion=diffusion, base_diffusion=base_diffusion)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    logger.log('load from: ' + args.model_path)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # timesteps = [
    #     [534, 834, 200, 400, 684],
    #     [24, 800, 573, 877, 200, 300, 400, 500, 621, 700],
    #     [644, 737, 67, 804, 134, 871, 6, 639, 268, 335, 402, 469, 536, 603, 670],
    #     'ddim5',
    #     'ddim10',
    #     'ddim15',
    # ]
    timesteps = [
        [534, 834, 200, 400, 684],
        [24, 800, 573, 877, 200, 300, 400, 500, 621, 700],
        [644, 737, 67, 804, 134, 871, 6, 639, 268, 335, 402, 469, 536, 603, 670],
        'ddim5',
        'ddim10',
        'ddim15',
    ]

    total_num = 12
    all_noise = th.randn(*(total_num, 3, args.image_size, args.image_size))
    all_classes = th.randint(
            low=0, high=NUM_CLASSES, size=(total_num,)
        )
    import copy
    for timestep in timesteps:
        if isinstance(timestep, str):
            diffusion_params = args_to_dict(args, diffusion_defaults().keys())
            diffusion_params['steps'] = diffusion_params['diffusion_steps']
            diffusion_params['timestep_respacing'] = timestep
            del diffusion_params['diffusion_steps']
            diffusion = create_gaussian_diffusion(
                **diffusion_params
            )
        
        else:
            diffusion_params = args_to_dict(args, diffusion_defaults().keys())
            diffusion_params['steps'] = diffusion_params['diffusion_steps']
            del diffusion_params['diffusion_steps']
            base_diffusion = create_gaussian_diffusion(
                **diffusion_params
            )
            diffusion = copy.deepcopy(base_diffusion)

            try:
                reset_diffusion(use_timesteps=timestep, active_diffusion=diffusion, base_diffusion=base_diffusion)
            except:
                import pdb
                pdb.set_trace()
                reset_diffusion(use_timesteps=timestep, active_diffusion=diffusion, base_diffusion=base_diffusion)

        logger.log("sampling...")

        all_images = []
        all_labels = []
        noise = copy.deepcopy(all_noise)
        noise = noise.to(dist_util.dev())

        t1 = time.time()

        

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=noise,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        sample = sample.cpu().numpy()

        shape_str = "x".join([str(x) for x in sample.shape])

        out_path = os.path.join(logger.get_dir(), str(timestep) + '_samples_'+str(shape_str)+'.npz')
        logger.log('saving to ' + str(out_path))
        np.savez(out_path, sample)

        del noise
        del diffusion

        sample_time = time.time() - t1
    
        dist.barrier()
        logger.log("sampling time: " + str(sample_time))
        logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_dir="",
        use_timestep=None,
        port='12345',
        gpu='',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
