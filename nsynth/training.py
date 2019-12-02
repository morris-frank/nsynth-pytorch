import os
import time
from statistics import mean
from typing import List, Dict

import torch
from torch import nn
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils import data

from .scheduler import ManualMultiStepLR


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


def train(model: nn.Module, gpu: List[int], trainset: data.DataLoader,
          testset: data.DataLoader, paths: Dict, iterpoints: Dict, n_it: int,
          use_board: bool, use_manual_scheduler: bool):
    """

    :param model: The WaveNet model Module
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
    # Move model to device(s):
    device = f'cuda:{gpu[0]}' if gpu else 'cpu'
    if gpu:
        model = nn.DataParallel(model.to(device), device_ids=gpu)

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), eps=1e-8, lr=2e-4)
    scheduler = _setup_scheduler(optimizer, use_manual_scheduler, n_it)

    cross_entropy = nn.CrossEntropyLoss()

    # Setup logging and save stuff
    if use_board:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    os.makedirs(paths['save'], exist_ok=True)
    save_path = f'{paths["save"]}/{{}}_NSynth.pt'
    losses = []
    iloader = iter(trainset)
    print_time = time.time()

    for it in range(n_it):
        # Load next random batch
        try:
            x, y = next(iloader)
        except StopIteration:
            iloader = iter(trainset)
            x, y = next(iloader)

        model.train()
        logits = model(x)
        loss = cross_entropy(logits, y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.detach().item())

        # LOG INFO
        if it % iterpoints['print'] == 0:
            mean_loss = mean(losses)
            losses = []
            mean_time = (time.time() - print_time) / iterpoints['print']
            print_time = time.time()
            print(f'it={it:>10}\tloss:{mean_loss:.3e}\t'
                  f'time/it:{mean_time}')
            if use_board:
                writer.add_scalar('Loss/train', mean_loss, it)
                writer.add_scalar('Mean Time', mean_time, it)

        # SAVE THE MODEL
        if it % iterpoints['save'] == 0:
            torch.save({
                'it': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path.format(it))

        # TEST THE MODEL
        if it % iterpoints['test'] == 0:
            test_losses = []
            model.eval()
            for x, y in testset:
                logits = model(x)
                loss = cross_entropy(logits, y.to(device))
                test_losses.append(loss.detach().item())

            mean_test_loss = mean(test_losses)
            print(f'TESTING: it={it:>10}\tloss:{mean_test_loss:.3e}\t')
            if use_board:
                writer.add_scalar('Loss/test', mean_test_loss, it)

    print(f'FINISH {n_it} mini-batches')
