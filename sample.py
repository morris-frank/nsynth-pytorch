import os
from os import path

import torch

from nsynth import WavenetVAE, WavenetAE, \
    make_config
from nsynth.sampling import generate, load_audio, load_model


def main(args):
    model_class = WavenetVAE if args.vae else WavenetAE
    device = f'cuda:{args.gpu[0]}' if args.gpu else 'cpu'
    print()

    # Build model
    model = model_class(bottleneck_dims=args.bottleneck_dims,
                        encoder_width=args.encoder_width,
                        decoder_width=args.decoder_width).to(device)

    model = load_model(args.weights, device, model)
    sample = load_audio(args.sample).to(device)

    print(f'Generate:\n'
          f'model: {model_class}\tsample: {sample.shape}')

    with torch.no_grad():
        generation, embedding = generate(model, sample)

    os.makedirs(args.sampledir, exist_ok=True)
    sp = f'{args.sampledir}/{path.splitext(path.basename(args.sample))[0]}' \
         f'_{model_class.__name__}.pt'
    torch.save({'generation': generation, 'embedding': embedding}, sp)


if __name__ == '__main__':
    main(make_config('sample').parse_args())
