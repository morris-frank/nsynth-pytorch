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
        q_scale = embedding[:, self.bottleneck_dims:, :]

        q = dist.Normal(q_loc, q_scale)
        x_q = q.rsample()

        output = self.decoder(x, x_q)
        return output, q, x_q

    @staticmethod
    def loss_function(model: nn.Module, x: torch.Tensor, targets: torch.Tensor):
        logits, q, x_q = model(x)

        ce_x = F.cross_entropy(logits, targets)
        zx_p_loc, zx_p_scale = torch.zeros(x_q.size()).cuda(), \
                               torch.ones(x_q.size()).cuda()
        pzx = dist.Normal(zx_p_loc, zx_p_scale)
        kl_zx = torch.sum(pzx.log_prob(x_q) - q.log_prob(x_q))

        loss = ce_x - kl_zx
        return logits, loss
