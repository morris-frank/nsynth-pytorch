from typing import Tuple

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from .decoder import WaveNetDecoder
from .encoder import TemporalEncoder
from .functional import shift1d
from .modules import AutoEncoder


class WavenetVAE(AutoEncoder):
    """
    The complete WaveNetAutoEncoder model.
    """

    def __init__(self, bottleneck_dims: int, encoder_width: int,
                 decoder_width: int, n_layers: int = 10, n_blocks: int = 3,
                 quantization_channels: int = 256,
                 channels: int = 1, gen: bool = False):
        """
        :param bottleneck_dims: Number of dims in the latent bottleneck.
        :param encoder_width: Width of the hidden layers in the encoder (Non-
            causal Temporal encoder).
        :param decoder_width: Width of the hidden layers in the decoder
            (WaveNet).
        :param n_layers: number of layers in encoder and decoder
        :param n_blocks: number of blocks in encoder and decoder
        :param quantization_channels:
        :param channels: Number of input channels.
        :param gen: Is this generation ?
        """
        super(WavenetVAE, self).__init__()
        self.bottleneck_dims = bottleneck_dims
        self.encoder = TemporalEncoder(bottleneck_dims=2 * bottleneck_dims,
                                       channels=channels, width=encoder_width,
                                       n_layers=n_layers, n_blocks=n_blocks)
        self.decoder = WaveNetDecoder(bottleneck_dims=bottleneck_dims,
                                      channels=channels, width=decoder_width,
                                      n_layers=n_layers, n_blocks=n_blocks,
                                      quantization_channels=quantization_channels,
                                      gen=gen)

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, dist.Normal, torch.Tensor]:
        embedding = self.encoder(x)
        q_loc = embedding[:, :self.bottleneck_dims, :]
        q_scale = F.softplus(embedding[:, self.bottleneck_dims:, :]) + 1e-7

        q = dist.Normal(q_loc, q_scale)
        x_q = q.rsample()
        x_q_log_prob = q.log_prob(x_q)

        x = shift1d(x, -1)
        logits = self.decoder(x, x_q)
        return logits, x_q, x_q_log_prob

    @staticmethod
    def loss_function(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, x_q, x_q_log_prob = model(x)

        ce_x = F.cross_entropy(logits, y.to(device))

        zx_p_loc = torch.zeros(x_q.size()).to(device)
        zx_p_scale = torch.ones(x_q.size()).to(device)
        pzx = dist.Normal(zx_p_loc, zx_p_scale)
        kl_zx = torch.sum(pzx.log_prob(x_q) - x_q_log_prob)

        loss = ce_x - kl_zx
        return logits, loss
