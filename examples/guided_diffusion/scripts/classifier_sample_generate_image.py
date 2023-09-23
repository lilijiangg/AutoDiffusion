"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    create_gaussian_diffusion,
)

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


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False, #False,
        rescale_learned_sigmas=False, #False,
    )

def main():
    import time
    t1 = time.time()
    args = create_argparser().parse_args()

    if args.use_mean:
        args.use_timestep = args.use_timestep.replace(' ', ',')
        args.use_timestep = eval(args.use_timestep)
        args.use_timestep = [round(args.use_timestep[i]) for i in range(len(args.use_timestep))]
        args.use_timestep = str(args.use_timestep)

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.MASTER_PORT

    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    dist_util.setup_dist()
    logger.configure(args.save_dir)

    logger.log(str(args))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    timesteps = [
        'ddim4',
        'ddim6',
        'ddim10',
        'ddim15',
        [153, 424, 926, 690],
        [123, 622, 207, 390, 948, 830],
        [15, 800, 100, 900, 200, 613, 447, 261, 972, 705],
        [0, 737, 67, 804, 134, 871, 5, 981, 268, 335, 402, 933, 536, 131, 670],
    ]
    

    total_num = 36

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
            # args.classifier_scale = 0.75
        
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
            # args.classifier_scale = 1.2

        logger.log("sampling...")
        
        model_kwargs = {}

        classes = copy.deepcopy(all_classes)
        classes = classes.to(dist_util.dev())
        noise = copy.deepcopy(all_noise)
        noise = noise.to(dist_util.dev())

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise=noise,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        sample = sample.cpu().numpy()
        labels = classes.cpu().numpy()

        shape_str = "x".join([str(x) for x in sample.shape])

        out_path = os.path.join(logger.get_dir(), str(timestep) + '_samples_'+str(shape_str)+'.npz')
        logger.log('saving to ' + str(out_path))
        np.savez(out_path, sample, labels)

        del noise
        del classes

        dist.barrier()
        t2 = time.time() - t1
        logger.log("sampling complete")
        logger.log("total time: " + str(t2))

    


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        save_dir="",
        classifier_scale=1.0,
        use_timestep=None,
        MASTER_PORT='12344',
        use_mean=False,
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
