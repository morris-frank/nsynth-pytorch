import os
from argparse import ArgumentParser
from datetime import datetime
from itertools import product
from statistics import mean

import torch
from torch import nn
from torch import optim
from torch.utils import data

from nsynth.autoencoder import WaveNetAutoencoder
from nsynth.data import NSynthDataset
from nsynth.scheduler import ManualMultiStepLR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--datadir', type=str, required=True,
                        help='The top-level directory of NSynth dataset '
                             '(containing the split directories.)')
    parser.add_argument('--savedir', type=str, default='./models/',
                        help='The path to save the checkpoints to.')
    parser.add_argument('--nbatch', type=int, default=32,
                        help='The batch size.')
    parser.add_argument('--nit', type=int, default=200000,
                        help='Number of batches to train for.')
    parser.add_argument('--itprint', type=int, default=1,
                        help='Frequency of loss print.')
    parser.add_argument('--itsave', type=int, default=1000,
                        help='Frequency of model checkpoints.')
    parser.add_argument('--ittest', type=int, default=1000,
                        help='Frequency of running the test set.')
    parser.add_argument('--bottleneck_dims', type=int, default=16,
                        help='Size ot the Autoencoder Bottleneck.')
    parser.add_argument('--encoder_width', type=int, default=128,
                        help='Dimensions of the encoders hidden layers.')
    parser.add_argument('--decoder_width', type=int, default=512,
                        help='Dimensions of the decoders hidden layers.')
    return parser.parse_args()


def main():
    args = parse_args()
    model = WaveNetAutoencoder(bottleneck_dims=args.bottleneck_dims,
                               encoder_width=args.encoder_width,
                               decoder_width=args.decoder_width,
                               channels=1).to(args.device)
    train(model, args.device, args.datadir, args.savedir, args.nbatch, args.nit,
          args.itprint, args.itsave, args.ittest)


def train(model: nn.Module, device: str, data_dir: str, save_dir: str,
          n_batch: int, n_it: int, it_print: int, it_save: int, it_test: int):
    # Settings for the LR Scheduler
    lr_milestones = [0, 90000, 120000, 150000, 180000, 210000, 240000]
    lr_gammas = [2e-4, 4e-4 / 3, 6e-5, 4e-5, 2e-5, 6e-6, 2e-6]

    save_dir = os.path.normpath(save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    start_time = f'{datetime.today():%Y-%m-%d_%H}'
    save_path = f'{save_dir}/{start_time}_NSynth.pt'

    optimizer = optim.Adam(model.parameters(), eps=1e-8)
    scheduler = ManualMultiStepLR(optimizer, lr_milestones, lr_gammas)

    train_set = NSynthDataset(data_dir, subset='train')
    print(train_set)
    n_epochs = (n_it * n_batch) // len(train_set)
    print(f'Training for {n_epochs} epochs.')

    test_set = NSynthDataset(data_dir, subset='test')

    loader = data.DataLoader(train_set, n_batch, shuffle=True)

    losses = []
    for it, (epoch, batch) in enumerate(product(range(n_epochs), loader)):
        model.train()
        loss = model(batch['audio'].to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if it % it_print == 0:
            print(f'it={it:>10},loss:{mean(losses[-it_print:]):.3e}')

        if it % it_save == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path)

        if it % it_test == 0:
            print('RUNNING TEST')
            raise NotImplementedError


if __name__ == '__main__':
    main()
