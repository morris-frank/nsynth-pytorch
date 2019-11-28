import os
import time
from datetime import datetime
from statistics import mean

import torch
from torch import nn
from torch import optim
from torch.utils import data

from nsynth import WaveNetAutoencoder, ManualMultiStepLR
from nsynth.config import make_config
from nsynth.data import AudioOnlyNSynthDataset


def main(args):
    device = f'cuda:{args.gpu[0]}' if args.gpu else 'cpu'
    model = WaveNetAutoencoder(bottleneck_dims=args.bottleneck_dims,
                               encoder_width=args.encoder_width,
                               decoder_width=args.decoder_width,
                               channels=1).to(device)
    if args.gpu:
        model = nn.DataParallel(model, device_ids=args.gpu)
    train(model, device, args.datadir, args.savedir, args.nbatch,
          args.nit, args.itprint, args.itsave, args.ittest, args.crop_length)


def train(model: nn.Module, device: str, data_dir: str, save_dir: str,
          n_batch: int, n_it: int, it_print: int, it_save: int, it_test: int,
          crop: int):
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

    train_set = AudioOnlyNSynthDataset(data_dir, subset='valid', crop=crop)
    loader = data.DataLoader(train_set, batch_size=n_batch, shuffle=True)

    # test_set = AudioOnlyNSynthDataset(data_dir, subset='test')

    cross_entropy = nn.CrossEntropyLoss()

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
            if it % it_print == 0:
                print(f'it={it:>10}\tloss:{mean(losses[-it_print:]):.3e}\t'
                      f'time/it:{(time.time() - epoch_time) / (it + 1)}')

            if False and it % it_save == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_path)

            if it % it_test == 0:
                print('RUNNING TEST')
                # raise NotImplementedError


if __name__ == '__main__':
    main(make_config().parse_args())
