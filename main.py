from torch.utils import data
from nsynth import WaveNetAutoencoder
from nsynth.config import make_config
from nsynth.training import train
from nsynth.data import AudioOnlyNSynthDataset


def main(args):
    # Build model
    model = WaveNetAutoencoder(bottleneck_dims=args.bottleneck_dims,
                               encoder_width=args.encoder_width,
                               decoder_width=args.decoder_width,
                               channels=1)

    # Build datasets
    dsets = dict()
    for subset in ['train', 'test']:
        dset = AudioOnlyNSynthDataset(root=args.datadir, subset=subset,
                                      families=args.families,
                                      sources=args.sources,
                                      crop=args.crop_length)
        dsets[subset] = data.DataLoader(dset, batch_size=args.nbatch,
                                        num_workers=8, shuffle=True)

    train(model=model,
          gpu=args.gpu,
          trainset=dsets['train'],
          testset=dsets['test'],
          paths={'save': args.savedir, 'log': args.logdir},
          iterpoints={'print': args.itprint, 'save': args.itsave,
                      'test': args.ittest},
          n_it=args.nit,
          use_board=args.board,
          use_manual_scheduler=args.original_lr_scheduler
          )


if __name__ == '__main__':
    main(make_config().parse_args())
