from nsynth import WaveNetAutoencoder, WaveNetVariationalAutoencoder,\
    make_config
from nsynth.training import train
from nsynth.data import make_loaders


def main(args):
    wavenet = WaveNetVariationalAutoencoder if args.vae else WaveNetAutoencoder
    # Build model
    model = wavenet(
        bottleneck_dims=args.bottleneck_dims,
        encoder_width=args.encoder_width, decoder_width=args.decoder_width)

    # Build datasets
    loaders = make_loaders(args.datadir, ['train', 'test'], args.nbatch,
                           args.crop_length, args.families, args.sources)

    train(model=model,
          loss_function=wavenet.loss_function,
          gpu=args.gpu,
          trainset=loaders['train'],
          testset=loaders['test'],
          paths={'save': args.savedir, 'log': args.logdir},
          iterpoints={'print': args.itprint, 'save': args.itsave,
                      'test': args.ittest},
          n_it=args.nit,
          use_board=args.board,
          use_manual_scheduler=args.original_lr_scheduler
          )


if __name__ == '__main__':
    main(make_config().parse_args())
