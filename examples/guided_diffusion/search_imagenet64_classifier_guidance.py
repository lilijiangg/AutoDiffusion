# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time

import argparse
import os

from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_classifier,
    classifier_defaults,
)

print = functools.partial(print, flush=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]  # [10, 20]
    size_per = num_timesteps // len(section_counts) # 500  1000
    extra = num_timesteps % len(section_counts)  # 0
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)  # 500
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)  # (500 - 1) / (10 - 1) = 55.4
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2
        import warnings
        from scipy import linalg
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class EvolutionSearcher(object):

    def __init__(self, args, model, base_diffusion, time_step, classifier, search_space=None):
        self.args = args
        self.model = model
        self.base_diffusion = base_diffusion
        self.classifier = classifier
        import copy
        self.active_diffusion = copy.deepcopy(base_diffusion)
        self.time_step = time_step
        # self.cfg = cfg
        ## EA hyperparameters
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        ## tracking variable 
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.max_fid = args.max_fid
        self.thres = args.thres
        
        # self.RandomForestClassifier = RandomForestClassifier(n_estimators=40)
        self.rf_features = []
        self.rf_lebal = []

        self.search_space = search_space

        from evaluations.evaluator_v1 import Evaluator_v1
        import tensorflow.compat.v1 as tf
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        self.evaluator = Evaluator_v1(tf.Session(config=config))

        import pickle
        f = open(args.ref_path,'rb')
        self.ref_stats = pickle.load(f)

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

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logger.log('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def sample_active_subnet(self):
        if self.search_space is not None:
            use_timestep = self.search_space
        else:
            original_num_steps = self.base_diffusion.original_num_steps
            use_timestep = [i for i in range(original_num_steps)]

        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step]
        return use_timestep
    
    def is_legal_before_search(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logger.log('cand: {} has visited!'.format(cand))
            return False
        info['fid'] = self.get_cand_fid(args=self.args, cand=eval(cand))
        logger.log('cand: {}, fid: {}'.format(cand, info['fid']))

        info['visited'] = True
        return True

    def is_legal(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logger.log('cand: {} has visited!'.format(cand))
            return False
        # if self.RandomForestClassifier.predict_proba(np.asarray(eval(cand), dtype='float')[None, :])[0,1] < self.thres: # 拒绝
        #     logger.log('cand: {} is not legal.'.format(cand))
        #     return False
        info['fid'] = self.get_cand_fid(args=self.args, cand=eval(cand))
        logger.log('cand: {}, fid: {}'.format(cand, info['fid']))

        # self.rf_features.append(eval(cand))
        # self.rf_lebal.append(info['fid'] < self.max_fid)

        info['visited'] = True
        return True

    def get_cand_fid(self, cand=None, args=None):
        # active model
        t1 = time.time()
        self.reset_diffusion(cand)
        reset_time = time.time() - t1
        t1 = time.time()
        
        self.model.eval()
        self.classifier.eval()

        # sample image
        def cond_fn(x, t, y=None):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = self.classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

        def model_fn(x, t, y=None):
            assert y is not None
            return self.model(x, t, y if args.class_cond else None)

        logger.log("sampling...")
        all_images = []
        all_labels = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            sample_fn = (
                self.active_diffusion.p_sample_loop if not args.use_ddim else self.active_diffusion.ddim_sample_loop
            )
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
        arr = arr[: args.num_samples]  # npz文件
        dist.barrier()
        logger.log("sampling complete")
        sample_time = time.time() - t1
        t1 = time.time()
        
        from evaluations.evaluator_v1 import cal_fid, FIDStatistics
        fid = cal_fid(arr, 64, self.evaluator, ref_stats=self.ref_stats)

        fid_time = time.time() - t1
        logger.log('reset_time: ' + str(reset_time) + ', sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))
        return fid

    def get_random_before_search(self, num):
        logger.log('random select ........')
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logger.log('random {}/{}'.format(len(self.candidates), num))
        logger.log('random_num = {}'.format(len(self.candidates)))
    
    def get_random(self, num):
        logger.log('random select ........')
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.log('random {}/{}'.format(len(self.candidates), num))
        logger.log('random_num = {}'.format(len(self.candidates)))

    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logger.log('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])

            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.log('cross {}/{}'.format(len(res), cross_num))

        logger.log('cross_num = {}'.format(len(res)))
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logger.log('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            if self.search_space is not None:
                all_index = self.search_space
            else:
                all_index = [j for j in range(self.base_diffusion.original_num_steps)]

            for i in all_index:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.log('mutation {}/{}'.format(len(res), mutation_num))

        logger.log('mutation_num = {}'.format(len(res)))
        return res
    
    def mutate_init_x(self, x0, mutation_num, m_prob):
        logger.log('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            if self.search_space is not None:
                all_index = self.search_space
            else:
                all_index = [k for k in range(self.base_diffusion.original_num_steps)]

            for i in all_index:
                if i not in cand:
                    candidates.append(i)
            
            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度 
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logger.log('mutation x0 {}/{}'.format(len(res), mutation_num))

        logger.log('mutation_num = {}'.format(len(res)))
        return res

    def search(self):
        logger.log('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        if args.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)

        else:
            steps = self.base_diffusion.original_num_steps
            if args.use_ddim:
                timestep_respacing = 'ddim'
            else:
                timestep_respacing = ''
            timestep_respacing += str(args.time_step)
            init_x = space_timesteps(steps, timestep_respacing)
            init_x = list(init_x)
            self.is_legal_before_search(str(init_x))
            self.candidates.append(str(init_x))
            self.get_random_before_search(self.population_num // 2 + 1)
            res = self.mutate_init_x(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            self.candidates += res
        
        # self.rf_features = [eval(self.candidates[j]) for j in range(len(self.candidates))]
        # rf_features = np.asarray(self.rf_features, dtype='float')

        def dec(fid):
            return fid < self.max_fid

        # try:    
        #     self.rf_lebal = [dec(self.vis_dict[self.candidates[j]]['fid']) for j in range(len(self.candidates))]
        #     self.RandomForestClassifier.fit(rf_features, self.rf_lebal)
        # except:
        #     import pdb
        #     pdb.set_trace()

        while self.epoch < self.max_epochs:
            logger.log('epoch = {}'.format(self.epoch))

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['fid'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['fid'])

            logger.log('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logger.log('No.{} {} fid = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['fid']))
            
            # sys.exit()
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num) #变异+杂交凑不足population size的部分重新随机采样

            # rf_features = np.asarray(self.rf_features, dtype='float')
            # self.RandomForestClassifier.fit(rf_features, self.rf_lebal) # refit

            self.epoch += 1

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_dir="",
        time_step=100,
        # search config
        seed = 0,
        deterministic = False,
        local_rank = 0,
        max_epochs = 20,
        select_num = 10,
        population_num = 50,
        m_prob = 0.1,
        crossover_num = 25,
        mutation_num = 35,
        classifier_path="",
        classifier_scale=1.0,
        max_fid=48.0,
        thres=0.2,
        use_ddim_init_x=False,
        search_space='',
        ref_path='',
        MASTER_PORT='12344',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == '__main__':

    args = create_argparser().parse_args()

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.MASTER_PORT

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())

    if args.search_space != '':
        search_space_core = sorted(eval(args.search_space))
        if args.use_ddim_init_x:
            steps = diffusion.original_num_steps
            if args.use_ddim:
                timestep_respacing = 'ddim'
            else:
                timestep_respacing = ''
            timestep_respacing += str(args.time_step)
            init_x = space_timesteps(steps, timestep_respacing)
            init_x = list(init_x)
            search_space_core += init_x

        R = int(diffusion.original_num_steps / 100)
        search_space = []

        for s in search_space_core:
            left = max(s - R, 0)
            right = min(s + R, diffusion.original_num_steps)
            search_space += [i for i in range(left, right)]
        
        search_space = sorted(list(set(search_space)))
        logger.log('search space: ' + str(search_space))
    else:
        search_space = None
    
    ## build EA 
    t = time.time()
    searcher = EvolutionSearcher(args, model=model, base_diffusion=diffusion, time_step=args.time_step, classifier=classifier, search_space=search_space)
    searcher.search()
    logger.log('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))
