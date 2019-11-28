import os
import time
from datetime import datetime
from statistics import mean
from typing import List

import torch
from torch import nn
from torch import optim
from torch.utils import data

from .data import AudioOnlyNSynthDataset
from .scheduler import ManualMultiStepLR


def train(model: nn.Module, gpu: List[int], data_dir: str, save_dir: str,
          crop: int, n_batch: int, n_it: int, it_print: int, it_save: int,
          it_test: int, use_board: bool):
    """

    :param model: The WaveNet model Module
    :param gpu: List of GPUs to use (int indexes)
    :param data_dir: Path to training data
    :param save_dir: Path to save the model to
    :param crop: Size of the random training sample crops
    :param n_batch: Size of mini-batch
    :param n_it: Number of iterations
    :param it_print: Frequency of printing stats
    :param it_save: Frequency of saving the model
    :param it_test: Frequency of testing the model
    :param use_board: Whether to use tensorboard
    :return:
    """
    # Move model to device(s):
    device = f'cuda:{gpu[0]}' if gpu else 'cpu'
    if gpu:
        model = nn.DataParallel(model.to(device), device_ids=gpu)

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), eps=1e-8)
    lr_milestones = [0, 90000, 120000, 150000, 180000, 210000, 240000]
    lr_gammas = [2e-4, 4e-4 / 3, 6e-5, 4e-5, 2e-5, 6e-6, 2e-6]
    scheduler = ManualMultiStepLR(optimizer, lr_milestones, lr_gammas)

    cross_entropy = nn.CrossEntropyLoss()

    # Setup data loaders for test and training set
    train_set = AudioOnlyNSynthDataset(data_dir, subset='valid', crop=crop)
    loader = data.DataLoader(train_set, batch_size=n_batch, shuffle=True)
    # test_set = AudioOnlyNSynthDataset(data_dir, subset='test')

    # Setup logging and save stuff
    if use_board:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{datetime.today():%Y-%m-%d_%H}_NSynth.pt'
    losses = []

    n_epochs = (n_it * n_batch) // len(train_set) + 1
    print(f'Training for {n_epochs} epochs with batch size {n_batch}.')
    for epoch in range(n_epochs):
        epoch_time = time.time()
        for it, (x, y) in enumerate(loader):
            model.train()
            logits = model(x)
            loss = cross_entropy(logits, y.to(device))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.detach().item())
            if use_board:
                writer.add_scalar('Loss/train', losses[-1], (epoch+1) * it)
            if it % it_print == 0:
                print(f'it={it:>10}\tloss:{mean(losses[-it_print:]):.3e}\t'
                      f'time/it:{(time.time() - epoch_time) / (it + 1)}')

            if it % it_save == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_path)

            if it % it_test == 0:
                print('RUNNING TEST')
                # raise NotImplementedError
