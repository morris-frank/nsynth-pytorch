import torch
from torch import nn

from .decoder import WaveNetDecoder
from .encoder import TemporalEncoder


class WaveNetAutoencoder(nn.Module):
    def __init__(self):
        super(WaveNetAutoencoder, self).__init__()
        self.encoder = TemporalEncoder()
        self.decoder = WaveNetDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        output = self.decoder(x, embedding)
        return output
