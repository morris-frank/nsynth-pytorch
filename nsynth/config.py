from argparse import ArgumentParser


def make_config() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True,
                        help='The top-level directory of NSynth dataset '
                             '(containing the split directories.)')

    args_train = parser.add_argument_group('Training options')
    args_train.add_argument('--gpu', type=int, required=False, nargs='+',
                            help='The GPU ids to use. If unset, will use CPU.')
    args_train.add_argument('--nit', type=int, default=200000,
                            help='Number of batches to train for.')
    args_train.add_argument('--nbatch', type=int, default=32,
                            help='The batch size.')
    args_train.add_argument('--crop_length', type=int, default=6144,
                            help='Length of the actual training sub-samples'
                                 '(crops).')

    args_log = parser.add_argument_group('Logging options')
    args_log.add_argument('--itprint', type=int, default=10,
                          help='Frequency of loss print.')
    args_log.add_argument('--itsave', type=int, default=1000,
                          help='Frequency of model checkpoints.')
    args_log.add_argument('--ittest', type=int, default=1000,
                          help='Frequency of running the test set.')
    args_log.add_argument('--savedir', type=str, default='./models/',
                          help='The path to save the checkpoints to.')

    args_model = parser.add_argument_group('Model options')
    args_model.add_argument('--bottleneck_dims', type=int, default=16,
                            help='Size ot the Autoencoder Bottleneck.')
    args_model.add_argument('--encoder_width', type=int, default=128,
                            help='Dimensions of the encoders hidden layers.')
    args_model.add_argument('--decoder_width', type=int, default=512,
                            help='Dimensions of the decoders hidden layers.')
    return parser
