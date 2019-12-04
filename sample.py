from nsynth import WavenetVAE, WavenetAE, \
    make_config
from nsynth.sampling import generate, load_audio, load_model


def main(args):
    model_class = WavenetVAE if args.vae else WavenetAE
    device = f'cuda:{args.gpu[0]}' if args.gpu else 'cpu'

    # Build model
    model = model_class(bottleneck_dims=args.bottleneck_dims,
                        encoder_width=args.encoder_width,
                        decoder_width=args.decoder_width)

    model = load_model(args.weights, device, model)
    sample = load_audio(args.sample)

    generation, embedding = generate(model, sample)


if __name__ == '__main__':
    main(make_config('sample').parse_args())
