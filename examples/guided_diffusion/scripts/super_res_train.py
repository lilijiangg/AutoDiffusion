"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F
import torch

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.save_dir)
    
    torch.cuda.set_device(args.local_rank)
    torch.cuda.empty_cache()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    
    if args.model_dir is not None:
        pretrain = dist_util.load_state_dict(args.model_dir, map_location="cpu")
        
        if pretrain['input_blocks.0.0.weight'].shape[1] == 3:
            pretrain['input_blocks.0.0.weight'] = torch.cat(
                [pretrain['input_blocks.0.0.weight'], 
                torch.zeros_like(pretrain['input_blocks.0.0.weight'])], 
                dim=1,
            )
            logger.log('input shape change:', pretrain['input_blocks.0.0.weight'].shape)
        if not args.learn_sigma and pretrain['out.2.weight'].shape[0] == 6:
            pretrain['out.2.weight'], _ = torch.split(pretrain['out.2.weight'], 3, dim=0)
            pretrain['out.2.bias'], _ = torch.split(pretrain['out.2.bias'], 3, dim=0)
            logger.log('output shape change:', pretrain['out.2.weight'].shape)
            
        model.load_state_dict(
            pretrain,
            strict=True,
        )
        logger.log(f'loading model from: {args.model_dir} complete.')
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        lq_paths=args.lq_paths,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    
def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False, lq_paths=None):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        random_crop=True,
        lq_paths=lq_paths,
        small_size=small_size,
    )
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_dir='',
        model_dir=None,
        lq_paths=None,
        local_rank=0,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
