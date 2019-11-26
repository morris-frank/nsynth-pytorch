import torch
from torch import nn

from .decoder import WaveNetDecoder
from .encoder import TemporalEncoder


class WaveNetAutoencoder(nn.Module):
    def __init__(self, bottleneck_dims: int, channels: int, encoder_width: int,
                 decoder_width: int):
        super(WaveNetAutoencoder, self).__init__()
        self.encoder = TemporalEncoder(bottleneck_dims=bottleneck_dims,
                                       channels=channels, width=encoder_width)
        self.decoder = WaveNetDecoder(bottleneck_dims=bottleneck_dims,
                                      channels=channels, width=decoder_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        output = self.decoder(x, embedding)
        return output
