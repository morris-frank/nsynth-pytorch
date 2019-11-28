from nsynth import WaveNetAutoencoder
from nsynth.config import make_config
from nsynth.training import train


def main(args):
    # Build model
    model = WaveNetAutoencoder(bottleneck_dims=args.bottleneck_dims,
                               encoder_width=args.encoder_width,
                               decoder_width=args.decoder_width,
                               channels=1)

    train(model=model, gpu=args.gpu, data_dir=args.datadir,
          save_dir=args.savedir, crop=args.crop_length, n_batch=args.nbatch,
          n_it=args.nit, it_print=args.itprint, it_save=args.itsave,
          it_test=args.ittest, use_board=args.board)


if __name__ == '__main__':
    main(make_config().parse_args())
