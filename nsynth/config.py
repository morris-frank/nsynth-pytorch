from argparse import ArgumentParser
from os import path
from . import WavenetAE, WavenetVAE
from .modules import AutoEncoder


def make_config(version: str) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, required=False, nargs='+',
                        help='The GPU ids to use. If unset, will use CPU.')

    if 'train' in version:
        parser.add_argument('--datadir', type=path.abspath, required=True,
                            help='The top-level directory of NSynth dataset '
                                 '(containing the split directories.)')

        gtrain = parser.add_argument_group('Training options')
        gtrain.add_argument('--nit', type=int, default=250000,
                            help='Number of batches to train for.')
        gtrain.add_argument('--nbatch', type=int, default=32,
                            help='The batch size.')
        gtrain.add_argument('--crop_length', type=int, default=6144,
                            help='Length of the actual training sub-samples'
                                 '(crops).')
        gtrain.add_argument('--original_lr_scheduler', action='store_true',
                            help='Use the original exact learning rate '
                                 'schedule as given in the paper.')
        gtrain.add_argument('--families', type=str, required=False, nargs='+',
                            help='The instrument families to use from the '
                                 'dataset.')
        gtrain.add_argument('--sources', type=str, required=False, nargs='+',
                            help='The instrument sources to use from the '
                                 'dataset.')

        glog = parser.add_argument_group('Logging options')
        glog.add_argument('--itprint', type=int, default=20,
                          help='Frequency of loss print.')
        glog.add_argument('--itsave', type=int, default=5000,
                          help='Frequency of model checkpoints.')
        glog.add_argument('--ittest', type=int, default=500,
                          help='Frequency of running the test set.')
        glog.add_argument('--savedir', type=path.abspath, default='./models/',
                          help='Path to save the checkpoints to.')
        glog.add_argument('--logdir', type=path.abspath, default='./log/',
                          help='Path to save the logs to.')
        glog.add_argument('--board', action='store_true',
                          help='Whether to use Tensorboard.')

    if 'sampl' in version:
        gsampl = parser.add_argument_group('Sampling options')
        gsampl.add_argument('--weights', type=path.abspath, required=True,
                            help='Path to the saved weight file.')
        gsampl.add_argument('--sample', type=path.abspath, required=True,
                            help='Path to the sample WAV file.')
        gsampl.add_argument('--sampledir', type=path.abspath,
                            default='./samples',
                            help='Path to save the generated samples to.')

    gmodel = parser.add_argument_group('Model options')
    gmodel.add_argument('--bottleneck_dims', type=int, default=16,
                        help='Size ot the Autoencoder Bottleneck.')
    gmodel.add_argument('--encoder_width', type=int, default=128,
                        help='Dimensions of the encoders hidden layers.')
    gmodel.add_argument('--decoder_width', type=int, default=512,
                        help='Dimensions of the decoders hidden layers.')
    gmodel.add_argument('--nlayers', type=int, default=10,
                        help='Number of dilation layers in each block.')
    gmodel.add_argument('--nblocks', type=int, default=3,
                        help='Number of blocks.')
    gmodel.add_argument('--vae', action='store_true',
                        help='Whether to use the VAE model.')
    return parser


def make_model(args) -> AutoEncoder:
    model_class = WavenetVAE if args.vae else WavenetAE

    args.decoder_gen = args.decoder_gen or False

    # Build model
    model = model_class(bottleneck_dims=args.bottleneck_dims,
                        encoder_width=args.encoder_width,
                        decoder_width=args.decoder_width,
                        n_layers=args.n_layers, n_blocks=args.n_blocks,
                        gen=args.decoder_gen)
    return model
