from argparse import ArgumentParser
from os import path


def make_config() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=path.abspath, required=True,
                        help='The top-level directory of NSynth dataset '
                             '(containing the split directories.)')

    args_train = parser.add_argument_group('Training options')
    args_train.add_argument('--gpu', type=int, required=False, nargs='+',
                            help='The GPU ids to use. If unset, will use CPU.')
    args_train.add_argument('--nit', type=int, default=250000,
                            help='Number of batches to train for.')
    args_train.add_argument('--nbatch', type=int, default=32,
                            help='The batch size.')
    args_train.add_argument('--crop_length', type=int, default=6144,
                            help='Length of the actual training sub-samples'
                                 '(crops).')
    args_train.add_argument('--original_lr_scheduler', action='store_true',
                            help='Use the original exact learning rate '
                                 'schedule as given in the paper.')
    args_train.add_argument('--families', type=str, required=False, nargs='+',
                            help='The instrument families to use from the '
                                 'dataset.')
    args_train.add_argument('--sources', type=str, required=False, nargs='+',
                            help='The instrument sources to use from the '
                                 'dataset.')

    args_log = parser.add_argument_group('Logging options')
    args_log.add_argument('--itprint', type=int, default=10,
                          help='Frequency of loss print.')
    args_log.add_argument('--itsave', type=int, default=1000,
                          help='Frequency of model checkpoints.')
    args_log.add_argument('--ittest', type=int, default=500,
                          help='Frequency of running the test set.')
    args_log.add_argument('--savedir', type=path.abspath, default='./models/',
                          help='The path to save the checkpoints to.')
    args_log.add_argument('--logdir', type=path.abspath, default='./log/',
                          help='The path to save the logs to.')
    args_log.add_argument('--board', action='store_true',
                          help='Whether to use Tensorboard.')

    args_model = parser.add_argument_group('Model options')
    args_model.add_argument('--bottleneck_dims', type=int, default=16,
                            help='Size ot the Autoencoder Bottleneck.')
    args_model.add_argument('--encoder_width', type=int, default=128,
                            help='Dimensions of the encoders hidden layers.')
    args_model.add_argument('--decoder_width', type=int, default=512,
                            help='Dimensions of the decoders hidden layers.')
    args_model.add_argument('--vae', action='store_true',
                            help='Whether to use the VAE model.')
    return parser
