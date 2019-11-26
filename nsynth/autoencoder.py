import torch
from torch import nn

from .decoder import WaveNetDecoder
from .encoder import TemporalEncoder


class WaveNetAutoencoder(nn.Module):
    """
    The complete WaveNetAutoEncoder model.
    """
    def __init__(self, bottleneck_dims: int, channels: int, encoder_width: int,
                 decoder_width: int):
        """
        :param bottleneck_dims: Number of dims in the latent bottleneck.
        :param channels: Number of input channels.
        :param encoder_width: Width of the hidden layers in the encoder (Non-
            causal Temporal encoder).
        :param decoder_width: Width of the hidden layers in the decoder
            (WaveNet).
        """
        super(WaveNetAutoencoder, self).__init__()
        self.encoder = TemporalEncoder(bottleneck_dims=bottleneck_dims,
                                       channels=channels, width=encoder_width)
        self.decoder = WaveNetDecoder(bottleneck_dims=bottleneck_dims,
                                      channels=channels, width=decoder_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        output = self.decoder(x, embedding)
        return output
