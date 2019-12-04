from collections import OrderedDict
from typing import Tuple

import librosa
import torch
from torch import nn
from tqdm import trange

from .functional import encode_μ_law
from .modules import AutoEncoder


def load_model(fp: str, device: str, model: nn.Module, train: bool = False) \
        -> nn.Module:
    """

    :param fp:
    :param device:
    :param model:
    :param train:
    :return:
    """
    save_point = torch.load(fp, map_location=torch.device(device))
    state_dict = save_point['model_state_dict']

    if next(iter(state_dict.keys())).startswith('module.'):
        _state_dict = OrderedDict({k[7:]: v for k, v in state_dict.items()})
        state_dict = _state_dict

    model.load_state_dict(state_dict)

    if not train:
        return model

    raise NotImplementedError


def load_audio(fp: str) -> torch.Tensor:
    """

    :param fp:
    :return:
    """
    raw, sr = librosa.load(fp, mono=True, sr=None)
    assert sr == 16000
    raw = torch.tensor(raw[None, None, ...], dtype=torch.float32)
    x = encode_μ_law(raw) / 128
    return x


def generate(model: AutoEncoder, x: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param model:
    :param x:
    :return:
    """

    model.eval()
    embedding = model.encoder(x)

    # Build and upsample all the conditionals from the embedding:
    l_upsample = nn.Upsample(scale_factor=model.decoder.scale_factor,
                             mode='nearest')
    l_conds = [l_upsample(l_cond(embedding))
               for l_cond in model.decoder.conds]
    l_conds.append(l_upsample(model.decoder.final_cond(embedding)))

    d_size = model.decoder.receptive_field
    generation = x[0, 0, :d_size]
    rem_length = x.numel() - d_size

    for _ in trange(rem_length):
        window = generation[-d_size:].view(1, 1, d_size)
        g_size = generation.numel() + 1
        conditionals = [l_conds[i][:, :, g_size - d_size:g_size]
                        for i in range(len(l_conds))]
        val = model.decoder(window, None, conditionals)[:, :, -1].squeeze()
        val = ((val.argmax().float() - 128.) / 128.).unsqueeze(0)
        generation = torch.cat((generation, val), 0)

    return generation.cpu(), embedding.cpu()
