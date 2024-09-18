# -*- coding: utf-8 -*-
"""
Created on 2024/9/18
@project: score_sde_pytorch
@filename: train
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description: 
"""
import gc
import io
import os
import time

import numpy as np
import logging
from simple_score import losses
from simple_score import sampling
from simple_score import sde_lib
from simple_score import utils
from simple_score.ema import ExponentialMovingAverage
import torch.optim as optim
import torch


from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, IterableDataset


def train(config, model, dataset, workdir, resume_path=None):


    # Initialize model and optimizer
    score_model = model
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = optim.Adam(score_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                                eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # data related
    dataset = dataset
    dataloader = DataLoader(dataset, config.training.batch_size)
    norm_func = utils.get_data_normalizer(config)
    inverse_func = utils.get_data_inverse_normalizer(config)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if resume_path is not None:
        # Resume training when intermediate checkpoints are detected
        state = utils.restore_checkpoint(resume_path, state, config.device)
    initial_step = int(state['step'])


    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_func, sampling_eps)

    num_train_steps = config.training.n_iters
    sum_loss = 0.0

    for step in range(initial_step, num_train_steps):
        batch = next(dataloader).to(config.device)
        batch = norm_func(batch)
        # Execute one training step
        loss = train_step_fn(state, batch)
        sum_loss += loss.item()
        if step % config.training.print_freq == 0:
            print(f"[{step}]loss={sum_loss/(1.0*config.training.print_freq)}")
            sum_loss = 0.0
        # Save a checkpoint periodically
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)



class SpiralDataset(IterableDataset):

    def __init__(self, centre=(0,0), r_min=1.0, r_max=5.0, rotation=10.0):
        super().__init__()
        self.centre = centre
        self.r_min = r_min
        self.r_max = r_max
        self.rotation = rotation

    def __iter__(self):
        while True:
            rotate = torch.rand(1)*self.rotation
            r = self.r_min + (self.r_max-self.r_min)*rotate/self.rotation
            x = self.centre[0] + r*torch.sin(rotate)
            y = self.centre[1] + r*torch.cos(rotate)
            data = torch.cat([x,y], dim=0)
            yield data


class SpiralListDataset(IterableDataset):

    def __init__(self, centre=(0,0), r_min=1.0, r_max=5.0, rotation=10.0, dr=0.2):
        super().__init__()
        self.centre = centre
        self.r_min = r_min
        self.r_max = r_max
        self.rotation = rotation
        self.dr = dr

    def __iter__(self):
        while True:
            rotate = torch.rand(1)*(self.rotation-3*self.dr) + torch.arange(start=0.0, end=2.5*self.dr, step=self.dr)
            r = self.r_min + (self.r_max-self.r_min)*rotate/self.rotation
            x = self.centre[0] + r*torch.sin(rotate)
            y = self.centre[1] + r*torch.cos(rotate)
            data = torch.stack([x,y], dim=-1).view(-1)
            yield data


