from typing import Tuple

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from .decoder import WaveNetDecoder
from .encoder import TemporalEncoder


class WaveNetVariationalAutoencoder(nn.Module):
    """
    The complete WaveNetAutoEncoder model.
    """

    def __init__(self, bottleneck_dims: int, encoder_width: int,
                 decoder_width: int, channels: int = 1):
        """
        :param bottleneck_dims: Number of dims in the latent bottleneck.
        :param encoder_width: Width of the hidden layers in the encoder (Non-
            causal Temporal encoder).
        :param decoder_width: Width of the hidden layers in the decoder
            (WaveNet).
        :param channels: Number of input channels.
        """
        super(WaveNetVariationalAutoencoder, self).__init__()
        self.bottleneck_dims = bottleneck_dims
        self.encoder = TemporalEncoder(bottleneck_dims=2 * bottleneck_dims,
                                       channels=channels, width=encoder_width)
        self.decoder = WaveNetDecoder(bottleneck_dims=bottleneck_dims,
                                      channels=channels, width=decoder_width)

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, dist.Normal, torch.Tensor]:
        embedding = self.encoder(x)
        q_loc = embedding[:, :self.bottleneck_dims, :]
        q_scale = F.softplus(embedding[:, self.bottleneck_dims:, :]) + 1e-7

        q = dist.Normal(q_loc, q_scale)
        x_q = q.rsample()
        x_q_log_prob = q.log_prob(x_q)

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
