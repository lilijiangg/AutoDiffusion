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
        # import pdb
        # pdb.set_trace()
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
    
    if args.use_timestep is not None:
        args.use_timestep = eval(args.use_timestep)
        args.use_timestep = sorted(args.use_timestep)
        import copy
        base_diffusion = copy.deepcopy(diffusion)
        reset_diffusion(use_timesteps=args.use_timestep, active_diffusion=diffusion, base_diffusion=base_diffusion)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    # import pdb
    # pdb.set_trace()
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.without_classifier:
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
            )
        else:
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log('created ' + str(len(all_images) * args.batch_size) + ' samples')

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), 'samples_'+str(shape_str)+'.npz')
        logger.log('saving to ' + str(out_path))
        np.savez(out_path, arr, label_arr)

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
        without_classifier=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
