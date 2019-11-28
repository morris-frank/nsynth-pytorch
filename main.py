from nsynth import WaveNetAutoencoder
from nsynth.config import make_config
from nsynth.training import train


def main(args):
    # Build model
    model = WaveNetAutoencoder(bottleneck_dims=args.bottleneck_dims,
                               encoder_width=args.encoder_width,
                               decoder_width=args.decoder_width,
                               channels=1)

    train(model, args.gpu, args.datadir, args.savedir, args.nbatch,
          args.nit, args.itprint, args.itsave, args.ittest, args.crop_length)


if __name__ == '__main__':
    main(make_config().parse_args())
