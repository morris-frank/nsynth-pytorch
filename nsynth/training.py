import os
import time
from datetime import datetime
from statistics import mean
from typing import List, Dict, Callable

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils import data

from .modules import AutoEncoder
from .scheduler import ManualMultiStepLR
from .visualization import ConfusionMatrix, log, MonkeyWriter


def _setup_scheduler(optimizer: Optimizer, use_manual_scheduler: bool,
                     n_it: int):
    if use_manual_scheduler:
        lr_milestones = [0, 90000, 120000, 150000, 180000, 210000, 240000]
        lr_gammas = [2e-4, 4e-4 / 3, 6e-5, 4e-5, 2e-5, 6e-6, 2e-6]
        scheduler = ManualMultiStepLR(optimizer, lr_milestones, lr_gammas)
    else:
        # For the automatic scheduler we approximate the manual scheduler
        # with a step LR based on the n_it:
        lr_milestones = torch.linspace(n_it * 0.36, n_it, 5).tolist()
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones,
                                                   gamma=0.6)
    return scheduler


def train(model: AutoEncoder, loss_function: Callable, gpu: List[int],
          trainset: data.DataLoader, testset: data.DataLoader, paths: Dict,
          iterpoints: Dict, n_it: int, use_board: bool,
          use_manual_scheduler: bool):
    """
    :param model: The WaveNet model Module
    :param loss_function: The static loss function, should take params:
        (model: Module, x: Tensor, y: Tensor, device: str)
    :param gpu: List of GPUs to use (int indexes)
    :param trainset: The dataset for training data
    :param testset: the dataset for testing data
    :param paths: The paths to save and  log to
    :param iterpoints: The number of iterations to print, save and test
    :param n_it: Number of iterations
    :param use_board: Whether to use tensorboard
    :param use_manual_scheduler: Whether to use the original manual scheduler
    :return:
    """
    # Setup logging and save stuff
    writer = MonkeyWriter()
    if use_board:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    os.makedirs(paths['save'], exist_ok=True)
    save_path = f'{paths["save"]}/{datetime.today():%y%m%d}_{{:06}}_' \
                f'{type(model).__name__}.pt'

    # Move model to device(s):
    device = f'cuda:{gpu[0]}' if gpu else 'cpu'
    if gpu:
        model = nn.DataParallel(model.to(device), device_ids=gpu)

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), eps=1e-8, lr=1e-3)
    scheduler = _setup_scheduler(optimizer, use_manual_scheduler, n_it)

    losses, it_times = [], []
    iloader = iter(trainset)
    for it in range(n_it):
        it_start_time = time.time()
        # Load next random batch
        try:
            x, y = next(iloader)
        except StopIteration:
            iloader = iter(trainset)
            x, y = next(iloader)

        model.train()
        logits, loss = loss_function(model, x, y, device)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(it)

        losses.append(loss.detach().item())
        it_times.append(time.time() - it_start_time)

        # LOG INFO
        if it % iterpoints['print'] == 0 or it == n_it - 1:
            log(writer, it, {'Loss/train': mean(losses),
                             'Time/train': mean(it_times),
                             'LR': optimizer.param_groups[0]['lr']})
            losses, it_times = [], []

        # SAVE THE MODEL
        if it % iterpoints['save'] == 0 or it == n_it - 1:
            torch.save({
                'it': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path.format(it))

        # TEST THE MODEL
        if it % iterpoints['test'] == 0 or it == n_it - 1:
            test_time, test_losses = time.time(), []
            conf_mat = ConfusionMatrix()

            model.eval()
            for x, y in testset:
                logits, loss = loss_function(model, x, y, device)
                test_losses.append(loss.detach().item())
                conf_mat.add(logits, y)

            np.save(f'./confusion_matrix_{it}.npy', conf_mat)

            log(writer, it, {'Loss/test': mean(test_losses),
                             'Class confusion': conf_mat.plot(),
                             'Time/test': time.time() - test_time})

    print(f'FINISH {n_it} mini-batches')
