from argparse import ArgumentParser

from torch import optim
from torch.utils import data

from nsynth.autoencoder import WaveNetAutoencoder
from nsynth.data import NSynthDataset
from nsynth.scheduler import ManualMultiStepLR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, nargs=1)
    parser.add_argument('--nbatch', type=int, default=32, nargs=1)
    parser.add_argument('--nit', type=int, default=200000, nargs=1)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.nbatch, args.nit)


def train(n_batch: int = 32, n_it: int = 200000):
    # Settings for Optimizer
    adam_eps = 1e-8
    # Settings for the LR Scheduler
    lr_milestones = [0, 90000, 120000, 150000, 180000, 210000, 240000]
    lr_gammas = [2e-4, 4e-4 / 3, 6e-5, 4e-5, 2e-5, 6e-6, 2e-6]

    model = WaveNetAutoencoder()
    optimizer = optim.Adam(model.parameters(), eps=adam_eps)
    scheduler = ManualMultiStepLR(optimizer, lr_milestones, lr_gammas)

    dataset = NSynthDataset()
    n_epochs = (n_it * n_batch) // len(dataset)

    loader = data.DataLoader(dataset, n_batch, shuffle=True)

    for epoch in range(n_epochs):
        for batch in loader:
            model.train()
            loss = model(batch['audio'])
            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


if __name__ == '__main__':
    main()
