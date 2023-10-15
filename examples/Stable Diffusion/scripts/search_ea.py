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
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os
from torch import autocast
from contextlib import contextmanager, nullcontext
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from scipy import linalg
from sklearn.ensemble import RandomForestClassifier

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.data.build_dataloader import build_dataloader

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import logging
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from pytorch_fid.inception import InceptionV3
import copy

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def get_activations(data, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):

    model.eval()

    if batch_size > data.shape[0]:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = data.shape[0]

    pred_arr = np.empty((data.shape[0], dims))
    start_idx = 0

    for i in range(0, data.shape[0], batch_size):
        if i + batch_size > data.shape[0]:
            batch = data[i:, :, :, :]
        else:
            batch = data[i:i+batch_size, :, :, :]
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]
    
    return pred_arr

def calculate_activation_statistics(datas, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    act = get_activations(datas, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(data1, ref_mu, ref_sigma, batch_size, device, dims, num_workers=1):
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(data1, model, batch_size,
                                            dims, device, num_workers)
    
    fid_value = calculate_frechet_distance(m1, s1, ref_mu, ref_sigma)

    return fid_value

class EvolutionSearcher(object):

    def __init__(self, opt, model, time_step, ref_mu, ref_sigma, sampler, dataloader_info, batch_size, dpm_params=None):
        self.opt = opt
        self.model = model
        self.sampler = sampler
        self.time_step = time_step
        self.dataloader_info = dataloader_info
        self.batch_size = batch_size
        # self.cfg = cfg
        ## EA hyperparameters
        self.max_epochs = opt.max_epochs
        self.select_num = opt.select_num
        self.population_num = opt.population_num
        self.m_prob = opt.m_prob
        self.crossover_num = opt.crossover_num
        self.mutation_num = opt.mutation_num
        self.num_samples = opt.num_sample
        self.ddim_discretize = "uniform"
        ## tracking variable 
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.max_fid = opt.max_fid
        # self.thres = opt.thres
        
        self.RandomForestClassifier = RandomForestClassifier(n_estimators=40)
        self.rf_features = []
        self.rf_lebal = []

        self.use_ddim_init_x = opt.use_ddim_init_x

        self.ref_mu = np.load(ref_mu)
        self.ref_sigma = np.load(ref_sigma)

        self.dpm_params = dpm_params
    
    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logging.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]
    
    def is_legal_before_search(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['fid'] = self.get_cand_fid(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, fid: {}'.format(cand, info['fid']))

        info['visited'] = True
        return True
    
    def is_legal(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        # if self.RandomForestClassifier.predict_proba(np.asarray(eval(cand), dtype='float')[None, :])[0,1] < self.thres: # 拒绝
        #     logging.info('cand: {} is not legal.'.format(cand))
        #     return False
        info['fid'] = self.get_cand_fid(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, fid: {}'.format(cand, info['fid']))

        info['visited'] = True
        return True
    
    def get_random_before_search(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_random(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
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
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res
    
    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
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
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def get_mutation_dpm(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
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
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def mutate_init_x(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
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
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
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
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def sample_active_subnet(self):
        original_num_steps = self.sampler.ddpm_num_timesteps
        use_timestep = [i for i in range(original_num_steps)]
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))] 
        return use_timestep
    
    def sample_active_subnet_dpm(self):
        use_timestep = copy.deepcopy(self.dpm_params['full_timesteps'])
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step + 1]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))] 
        return use_timestep
    
    def get_cand_fid(self, cand=None, opt=None, device='cuda'):
        t1 = time.time()
        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
 
        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    t1 = time.time()
                    all_samples = list()
                    for itr, batch in enumerate(self.dataloader_info['validation_loader']):
                        # for k, v in batch.items():
                        #     if torch.is_tensor(v):
                        #         batch[k] = v.cuda()
                        prompts = batch['text']
                        uc = None
                        if opt.scale != 1.0:
                            uc = self.model.get_learned_conditioning(self.batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        sampled_timestep = np.array(cand)
                        samples_ddim, _ = self.sampler.sample(S=opt.time_step,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code,
                                                         sampled_timestep=sampled_timestep)
                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                        x_checked_image = x_samples_ddim

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        for x_sample in x_checked_image_torch:
                            all_samples.append(x_sample.cpu().numpy())

                        logging.info('samples: ' + str(len(all_samples)))
                    
                        if len(all_samples) > self.num_samples:
                            logging.info('samples: ' + str(len(all_samples)))
                            break
        sample_time = time.time() - t1
        # active model
        t1 = time.time()
        all_samples = np.array(all_samples)
        all_samples = torch.Tensor(all_samples)
        fid = calculate_fid(data1=all_samples,ref_mu=self.ref_mu, ref_sigma=self.ref_sigma, batch_size=320, dims=2048, device='cuda')
        logging.info('FID: ' + str(fid))

        fid_time = time.time() - t1
        logging.info('sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))
        return fid
    
    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        if self.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)

        else:
            if self.opt.dpm_solver:
                init_x = self.dpm_params['init_timesteps']
            else:
                init_x = make_ddim_timesteps(ddim_discr_method=self.ddim_discretize, num_ddim_timesteps=self.time_step,
                                                        num_ddpm_timesteps=self.sampler.ddpm_num_timesteps,verbose=False)
            init_x = sorted(list(init_x))
            self.is_legal_before_search(str(init_x))
            self.candidates.append(str(init_x))
            self.get_random_before_search(self.population_num // 2)
            if self.opt.dpm_solver:
                res = self.mutate_init_x_dpm(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            else:
                res = self.mutate_init_x(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            self.candidates += res
        
        # self.rf_features = [eval(self.candidates[j]) for j in range(len(self.candidates))]
        # rf_features = np.asarray(self.rf_features, dtype='float')

        # try:    
        #     self.rf_lebal = [dec(self.vis_dict[self.candidates[j]]['fid']) for j in range(len(self.candidates))]
        #     self.RandomForestClassifier.fit(rf_features, self.rf_lebal)
        # except: 
        #     import pdb
        #     pdb.set_trace()

        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['fid'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['fid'])

            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logging.info('No.{} {} fid = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['fid']))
            
            if self.epoch + 1 == self.max_epochs:
                break
            # sys.exit()
            if self.opt.dpm_solver:
                mutation = self.get_mutation_dpm(
                    self.select_num, self.mutation_num, self.m_prob)
            else:
                mutation = self.get_mutation(
                    self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num) #变异+杂交凑不足population size的部分重新随机采样

            # rf_features = np.asarray(self.rf_features, dtype='float')
            # self.RandomForestClassifier.fit(rf_features, self.rf_lebal) # refit

            self.epoch += 1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="path to data",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=4,
        help="samples num",
    )
    parser.add_argument(
        "--cal_fid",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--select_num",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--population_num",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--m_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--crossover_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--mutation_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max_fid",
        type=float,
        default=3.,
    )
    parser.add_argument(
        "--thres",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--ref_mu",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ref_sigma",
        type=str,
        default='',
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--use_ddim_init_x",
        type=bool,
        default=False,
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")  # 加载模型

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)  # 采样器
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    dataloader_info = build_dataloader(config, opt)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # log
    os.makedirs(outpath, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(outpath, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    batch_size = opt.n_samples

    if opt.dpm_solver:
        tmp_sampler = DPMSolverSampler(model)
        from ldm.models.diffusion.dpm_solver.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
        ns = NoiseScheduleVP('discrete', alphas_cumprod=tmp_sampler.alphas_cumprod)
        dpm_solver = DPM_Solver(None, ns, predict_x0=True, thresholding=False)
        skip_type = "time_uniform"
        t_0 = 1. / dpm_solver.noise_schedule.total_N  # 0.001
        t_T = dpm_solver.noise_schedule.T  # 1.0
        full_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=1000, device='cpu')
        init_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=opt.time_step, device='cpu')
        dpm_params = dict()
        full_timesteps = list(full_timesteps)
        dpm_params['full_timesteps'] = [full_timesteps[i].item() for i in range(len(full_timesteps))]
        init_timesteps = list(init_timesteps)
        dpm_params['init_timesteps'] = [init_timesteps[i].item() for i in range(len(init_timesteps))]
    else:
        dpm_params = None


    ## build EA
    t = time.time()
    searcher = EvolutionSearcher(opt=opt, model=model, time_step=opt.time_step, ref_mu=opt.ref_mu, ref_sigma=opt.ref_sigma, sampler=sampler, dataloader_info=dataloader_info, batch_size=batch_size, dpm_params=dpm_params)
    searcher.search()
    logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

if __name__ == '__main__':
    main()
