import torch
from torch import distributions as dist
from torch.nn import functional as F

from .autoencoder import WaveNetAutoencoder
from .decoder import WaveNetDecoder
from .encoder import TemporalEncoder


class WaveNetVariationalAutoencoder(WaveNetAutoencoder):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        q_loc = embedding[:, :self.bottleneck_dims, :]
        q_scale = embedding[:, self.bottleneck_dims:, :]

        self.q = dist.Normal(q_loc, q_scale)
        self.x_q = self.q.rsample()

        output = self.decoder(x, embedding)
        return output

    def loss_function(self, logits: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        CE_x = F.cross_entropy(logits, targets)
        zx_p_loc, zx_p_scale = torch.zeros(logits.size()[0],
                                           self.bottleneck_dims).cuda(), \
                               torch.ones(logits.size()[0],
                                          self.bottleneck_dims).cuda()
        pzx = dist.Normal(zx_p_loc, zx_p_scale)
        KL_zx = torch.sum(pzx.log_prob(self.x_q) - self.q.log_prob(self.x_q))
        return CE_x - KL_zx
