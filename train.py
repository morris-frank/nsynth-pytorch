from nsynth import WavenetAE, WavenetVAE, \
    make_config
from nsynth.data import make_loaders
from nsynth.training import train
from nsynth.config import make_model


def main(args):
    model_class = WavenetVAE if args.vae else WavenetAE

    model = make_model(args)

    # Build datasets
    loaders = make_loaders(args.datadir, ['train', 'test'], args.nbatch,
                           args.crop_length, args.families, args.sources)
    train(model=model,
          loss_function=model_class.loss_function,
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
    main(make_config('train').parse_args())
