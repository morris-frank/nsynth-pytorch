import numpy as np
import torch
from torch import dtype as torch_dtype
from torch.utils import data

from nsynth import make_config, WavenetAE, WavenetVAE
from nsynth.training import train


class ToyDataSet(data.Dataset):
    def __init__(self, filepath: str, dtype: torch_dtype):
        self.data = np.load(filepath, allow_pickle=True)
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f'ToyDataSet <{len(self):>7} signals>'

    def __getitem__(self, idx: int):
        item = self.data[idx]
        mix = torch.tensor(item['mix'], dtype=self.dtype)
        sources = torch.tensor(item['sources'], dtype=self.dtype)
        return mix, sources


def main(args):
    model_class = WavenetVAE if args.vae else WavenetAE

    model = model_class(bottleneck_dims=args.bottleneck_dims,
                        encoder_width=args.encoder_width,
                        decoder_width=args.decoder_width)

    dset = ToyDataSet()
    loader = data.DataLoader(dset, batch_size=args.nbatch, num_workers=8,
                             shuffle=True)

    train(model=model,
          loss_function=model_class.loss_function,
          gpu=args.gpu,
          trainset=loader,
          testset=loader,
          paths={'save': './models_toy/', 'log': './log_toy/'},
          iterpoints={'print': args.itprint, 'save': args.itsave,
                      'test': args.ittest},
          n_it=args.nit,
          use_board=args.board,
          use_manual_scheduler=args.original_lr_scheduler
          )


if __name__ == '__main__':
    main(make_config('train').parse_args())
