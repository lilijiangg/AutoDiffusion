from cmath import isnan
import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tensorboardX import SummaryWriter

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from guided_diffusion.script_util import create_gaussian_diffusion
from guided_diffusion.resample import create_named_schedule_sampler
import numpy as np

import random
random.seed(0)

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        tensorboard_path=None
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler # or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        # self.use_ddp = False
        # self.ddp_model = self.model
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.tensorboard_path = tensorboard_path
        self.writer = None
        if self.tensorboard_path is not None:
            self.writer = SummaryWriter(os.path.join(self.tensorboard_path, 'log'))

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)  # batch: torch.Size([4, 3, 32, 32]), cond: {'y': tensor([2, 5, 8, 3])}
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev()) # 采样 t , 随机采样, weight表示扩散过程每个时刻的权重

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()  # 得到了返回的损失，包括l1、kl散度和两者的和
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            loss = (losses["loss"] * weights).mean()
            try:
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}, writer=self.writer, iter=self.step
                )
            except:
                import pdb
                pdb.set_trace()
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}, writer=self.writer, iter=self.step
                )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

class TrainValLoop(TrainLoop):
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        tensorboard_path=None,
        valid_frequence=10000,
        valid_batch_size = 128,
        valid_sample_timestep = 10,
        valid_diffusion = None,
        valid_num_sample = 1000,
        cond = False,
        use_ddim = False,
        image_size = 256,
        clip_denoised = None,
    ):
        super().__init__(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
        tensorboard_path=tensorboard_path)

        self.valid_frequence = valid_frequence
        self.valid_batch_size = valid_batch_size
        self.valid_sample_timestep = valid_sample_timestep
        self.valid_diffusion = valid_diffusion
        self.valid_num_sample = valid_num_sample
        self.cond = cond
        self.use_ddim = use_ddim
        self.image_size = image_size
        self.clip_denoised = clip_denoised

        from evaluations.evaluator_v1 import Evaluator_v1
        import tensorflow.compat.v1 as tf
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        self.evaluator = Evaluator_v1(tf.Session(config=config))

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)  # batch: torch.Size([4, 3, 32, 32]), cond: {'y': tensor([2, 5, 8, 3])}
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step % self.valid_frequence == 0:
                self.valid_step(writer=self.writer, iter=self.step)
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def valid_step(self, writer=None, iter=0):
        logger.log("sampling...")
        all_images = []
        all_labels = []
        while len(all_images) * self.valid_batch_size < self.valid_num_sample:
            model_kwargs = {}
            if self.cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(self.valid_batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                self.valid_diffusion.p_sample_loop if not self.use_ddim else self.valid_diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                (self.valid_batch_size, 3, self.image_size, self.image_size),
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if self.cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * self.valid_batch_size} samples")
        import numpy as np
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: self.valid_num_sample]  # npz文件
        dist.barrier()
        logger.log("sampling complete")

        from evaluations.evaluator_v1 import cal_fid, FIDStatistics
        fid = cal_fid(arr, 64, self.evaluator)
        logger.logkv('fid', fid)
        self.log_step()
        writer.add_scalar('fid', fid, iter)

class OFA_TrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        split_part1 = [250],
        split_part2 = [250],
        split_part3 = [250],
        split_part4 = [250],
        model_config = None,
        schedule_sampler_type = None,
        tensorboard_path=None,
    ):
        super().__init__(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
        tensorboard_path=tensorboard_path)

        self.diffusion = None
        self.schedule_sampler = None
        self.split_part1 = split_part1
        self.split_part2 = split_part2
        self.split_part3 = split_part3
        self.split_part4 = split_part4

        self.active_p1 = 0
        self.active_p2 = 0
        self.active_p3 = 0
        self.active_p4 = 0

        self.model_config = model_config
        self.schedule_sampler_type = schedule_sampler_type
    
    def forward_backward(self, batch, cond):
        # sample diffusion
        self.active_p1 = random.choice(self.split_part1)
        while self.active_p2 < self.active_p1:
            self.active_p2 = random.choice(self.split_part2)

        self.active_p4 = random.choice(self.split_part4)
        while self.active_p3 < self.active_p4:
            self.active_p3 = random.choice(self.split_part3)

        timestep_respacing = str(self.active_p1) + ',' + str(self.active_p2) + ',' \
                            + str(self.active_p3) + ',' + str(self.active_p4)
        self.active_diffusion = create_gaussian_diffusion(
            steps=self.model_config['diffusion_steps'],
            learn_sigma=self.model_config['learn_sigma'],
            noise_schedule=self.model_config['noise_schedule'],
            use_kl=self.model_config['use_kl'],
            predict_xstart=self.model_config['predict_xstart'],
            rescale_timesteps=self.model_config['rescale_timesteps'],
            rescale_learned_sigmas=self.model_config['rescale_learned_sigmas'],
            timestep_respacing=timestep_respacing
        )
        self.active_schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.active_diffusion)

        # train
        # 训练 1000 步 DDPM，再跑边采样边训练；
        # 采样 DDPM ， 
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.active_schedule_sampler.sample(micro.shape[0], dist_util.dev()) # 采样 t , 随机采样, weight表示扩散过程每个时刻的权重

            compute_losses = functools.partial(
                self.active_diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()  # 得到了返回的损失，包括l1、kl散度和两者的和
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.active_schedule_sampler, LossAwareSampler):
                self.active_schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.active_diffusion, t, {k: v * weights for k, v in losses.items()}, writer=self.writer, iter=self.step
            )
            self.mp_trainer.backward(loss)
            logger.logkv("sample_split", timestep_respacing)

class OFA_TrainLoop_random_select(TrainLoop):
    def __init__(
        self,
        *,
        model,
        base_diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        model_config = None,
        schedule_sampler_type = None,
        tensorboard_path=None,
        max_sample_step=400
    ):
        super().__init__(
        model=model,
        diffusion=base_diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
        tensorboard_path=tensorboard_path)

        self.base_diffusion = base_diffusion
        self.active_diffusion = copy.deepcopy(base_diffusion)
        self.schedule_sampler = None

        self.model_config = model_config
        self.schedule_sampler_type = schedule_sampler_type
        self.max_sample_step = max_sample_step

    def reset_diffusion(self, use_timesteps):
        use_timesteps = set(use_timesteps)
        self.active_diffusion.timestep_map = []
        last_alpha_cumprod = 1.0
        new_betas = []

        self.active_diffusion.use_timesteps = set(use_timesteps)

        for i, alpha_cumprod in enumerate(self.base_diffusion.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod) # 通过长序列的 \overline{alpha} 解 短序列的 \beta
                last_alpha_cumprod = alpha_cumprod
                self.active_diffusion.timestep_map.append(i)

        import numpy as np
        new_betas = np.array(new_betas, dtype=np.float64)

        self.active_diffusion.betas = new_betas
        assert len(new_betas.shape) == 1, "betas must be 1-D"
        assert (new_betas > 0).all() and (new_betas <= 1).all()

        self.active_diffusion.num_timesteps = int(new_betas.shape[0])

        alphas = 1.0 - new_betas  # alpha 递减
        self.active_diffusion.alphas_cumprod = np.cumprod(alphas, axis=0) # overliane_{x}
        self.active_diffusion.alphas_cumprod_prev = np.append(1.0, self.active_diffusion.alphas_cumprod[:-1]) # alpha[0], alpha[0], alpha[1], ...., alpha[T-1]
        self.active_diffusion.alphas_cumprod_next = np.append(self.active_diffusion.alphas_cumprod[1:], 0.0)  # alpha[1], alpha[2], ..., alpha[T], alpha[T]
        assert self.active_diffusion.alphas_cumprod_prev.shape == (self.active_diffusion.num_timesteps,)  

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.active_diffusion.sqrt_alphas_cumprod = np.sqrt(self.active_diffusion.alphas_cumprod)  # \sqrt{\overline{\alpha}}
        self.active_diffusion.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.active_diffusion.alphas_cumprod) # \sqrt{1 - \overline{\alpha}}
        self.active_diffusion.log_one_minus_alphas_cumprod = np.log(1.0 - self.active_diffusion.alphas_cumprod)  # \log{1 - \overline{\alpha}}
        self.active_diffusion.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.active_diffusion.alphas_cumprod)      # \frac{1}{\sqrt{\overline{\alpha}}}
        self.active_diffusion.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.active_diffusion.alphas_cumprod - 1)   

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.active_diffusion.posterior_variance = (
            new_betas * (1.0 - self.active_diffusion.alphas_cumprod_prev) / (1.0 - self.active_diffusion.alphas_cumprod)  # DDPM 式7 的 \hat{\beta}
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        if len(self.active_diffusion.posterior_variance) > 1:
            self.active_diffusion.posterior_log_variance_clipped = np.log(
                np.append(self.active_diffusion.posterior_variance[1], self.active_diffusion.posterior_variance[1:])
            )
        else:
            self.active_diffusion.posterior_log_variance_clipped = self.active_diffusion.posterior_variance
        self.active_diffusion.posterior_mean_coef1 = (
            new_betas * np.sqrt(self.active_diffusion.alphas_cumprod_prev) / (1.0 - self.active_diffusion.alphas_cumprod)
        )
        self.active_diffusion.posterior_mean_coef2 = (
            (1.0 - self.active_diffusion.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.active_diffusion.alphas_cumprod)
        )

    def forward_backward(self, batch, cond):
        # sample diffusion

        all_timesteps = []
        all_timesteps.append(['largest', [i for i in range(self.base_diffusion.original_num_steps)]])

        for subnet in range(2):
            timesteps = []
            original_num_steps = self.base_diffusion.original_num_steps
            
            num_timestep = random.randint(1, self.max_sample_step + 1)
            mapping_index = [1] * num_timestep + [0] * (self.max_sample_step - num_timestep)
            random.shuffle(mapping_index)
            skip = int(original_num_steps / len(mapping_index))
            for i in range(len(mapping_index)):
                if mapping_index[i] == 1:
                    timestep = random.randint(i * skip, (i + 1) * skip)
                    timesteps.append(timestep)
            # timesteps.append(self.base_diffusion.original_num_steps-1)
            # timesteps = set(timesteps)

            all_timesteps.append(['rand_'+str(subnet), timesteps])
        
        min_choise1 = random.randint(1, self.base_diffusion.original_num_steps-2)
        min_choise2 = random.randint(1, self.base_diffusion.original_num_steps-2)
        min_choise3 = random.randint(1, self.base_diffusion.original_num_steps-2)
        all_timesteps.append(['smallest', set([min_choise1, min_choise2, min_choise3, self.base_diffusion.original_num_steps-1])])

        # train
        self.mp_trainer.zero_grad()
        for name, active_timestep in all_timesteps:
            self.reset_diffusion(active_timestep)
            self.active_schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.active_diffusion)

            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i : i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.active_schedule_sampler.sample(micro.shape[0], dist_util.dev()) # 采样 t , 随机采样, weight表示扩散过程每个时刻的权重

                compute_losses = functools.partial(
                    self.active_diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )
                if last_batch or not self.use_ddp:
                    losses = compute_losses()  # 得到了返回的损失，包括l1、kl散度和两者的和
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                # if name == 'smallest':
                #     losses['loss'] = losses['mse'] * 2

                if isinstance(self.active_schedule_sampler, LossAwareSampler):
                    self.active_schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                loss = (losses["loss"] * weights).mean()
                log_loss_dict(
                    self.active_diffusion, t, {k: v * weights for k, v in losses.items()}, writer=self.writer, iter=self.step,
                    name='_'+name
                )
                import math
                if math.isnan(loss):
                    import pdb
                    pdb.set_trace()
                self.mp_trainer.backward(loss)
                logger.logkv("diffusion_len_"+name, len(active_timestep))

class OFA_TrainValLoop_kd(TrainValLoop):
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        tensorboard_path=None,
        valid_frequence=10000,
        valid_batch_size = 128,
        valid_sample_timestep = 10,
        valid_diffusion = None,
        valid_num_sample = 1000,
        cond = False,
        use_ddim = False,
        image_size = 256,
        clip_denoised = None,
        teacher_model = None,
    ):
        super().__init__(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
        tensorboard_path=tensorboard_path,
        valid_frequence=valid_frequence,
        valid_batch_size=valid_batch_size,
        valid_sample_timestep=valid_sample_timestep,
        valid_diffusion=valid_diffusion,
        valid_num_sample=valid_num_sample,
        cond=cond,
        use_ddim=use_ddim,
        image_size=image_size,
        clip_denoised=clip_denoised)

        self.teacher_model = teacher_model

    def kd_loss(self, x0, t, xt, model_kwargs):
        pass


    

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, writer=None, iter=0, name=''):
    try:
        for key, values in losses.items():
            logger.logkv_mean(key+name, values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            loss_list = [[], [], [], []]
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}"+name, sub_loss)
                loss_list[quartile].append(sub_loss)
            if writer is not None and iter % 1000 == 0:
                writer.add_scalar(key+name, values.mean(), iter)
                if name != 'smallest':
                    loss_list_mean0 = sum(loss_list[0]) / len(loss_list[0]) if len(loss_list[0]) > 0 else 0
                    loss_list_mean1 = sum(loss_list[1]) / len(loss_list[1]) if len(loss_list[1]) > 0 else 0
                    loss_list_mean2 = sum(loss_list[2]) / len(loss_list[2]) if len(loss_list[2]) > 0 else 0
                    loss_list_mean3 = sum(loss_list[3]) / len(loss_list[3]) if len(loss_list[3]) > 0 else 0
                    writer.add_scalar(key+'q_0'+name, loss_list_mean0, iter)
                    writer.add_scalar(key+'q_1'+name, loss_list_mean1, iter)
                    writer.add_scalar(key+'q_2'+name, loss_list_mean2, iter)
                    writer.add_scalar(key+'q_3'+name, loss_list_mean3, iter)
    except:
        import pdb
        pdb.set_trace()
        for key, values in losses.items():
            logger.logkv_mean(key+name, values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            loss_list = [[], [], [], []]
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}"+name, sub_loss)
                loss_list[quartile].append(sub_loss)
            if writer is not None and iter % 1000 == 0:
                writer.add_scalar(key+name, values.mean(), iter)
                if name != 'smallest':
                    loss_list_mean0 = sum(loss_list[0]) / len(loss_list[0]) if len(loss_list[0]) > 0 else 0
                    loss_list_mean1 = sum(loss_list[1]) / len(loss_list[1]) if len(loss_list[1]) > 0 else 0
                    loss_list_mean2 = sum(loss_list[2]) / len(loss_list[2]) if len(loss_list[2]) > 0 else 0
                    loss_list_mean3 = sum(loss_list[3]) / len(loss_list[3]) if len(loss_list[3]) > 0 else 0
                    writer.add_scalar(key+'q_0'+name, loss_list_mean0, iter)
                    writer.add_scalar(key+'q_1'+name, loss_list_mean1, iter)
                    writer.add_scalar(key+'q_2'+name, loss_list_mean2, iter)
                    writer.add_scalar(key+'q_3'+name, loss_list_mean3, iter)
