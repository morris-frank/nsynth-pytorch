import os
from os import path

import torch

from nsynth import WavenetVAE, WavenetAE, \
    make_config
from nsynth.config import make_model
from nsynth.sampling import generate, load_audio


def main(args):
    model_class = WavenetVAE if args.vae else WavenetAE
    device = f'cuda:{args.gpu[0]}' if args.gpu else 'cpu'

    args.decoder_gen = True
    model = make_model(args).to(device)

    # model = load_model(args.weights, device, model)

    d_size = model.decoder.receptive_field
    sample = load_audio(args.sample)
    numel = sample.numel()
    sample = sample[0, 0, :d_size].view(1, 1, d_size).to(device)

    with torch.no_grad():
        generation, embedding = generate(model, sample, numel, device)

    os.makedirs(args.sampledir, exist_ok=True)
    sp = f'{args.sampledir}/{path.splitext(path.basename(args.sample))[0]}' \
         f'_{model_class.__name__}.pt'
    torch.save({'generation': generation, 'embedding': embedding}, sp)


if __name__ == '__main__':
    main(make_config('sample').parse_args())
