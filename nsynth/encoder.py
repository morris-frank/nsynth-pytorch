from itertools import product

import torch
from torch import nn

from .modules import BlockWiseConv1d


class TemporalEncoder(nn.Module):
    """
    The Non-Causal Temporal Encoder as described in original NSynth
    [http://arxiv.org/abs/1704.01279].
    """

    def __init__(self,
                 channels: int = 1,
                 n_layers: int = 10,
                 n_blocks: int = 3,
                 width: int = 128,
                 kernel_size: int = 3,
                 bottleneck_dims: int = 16,
                 hop_length: int = 512):
        """
        :param channels: Number of input channels
        :param n_layers: Number of layers in each stage in the encoder
        :param n_blocks: Number of stages
        :param width: Size of the hidden channels in all layers
        :param kernel_size: KS for all 1D-convolutions
        :param bottleneck_dims: Final number of features
        :param hop_length: Final bottleneck pooling
        """
        super(TemporalEncoder, self).__init__()

        self.init = BlockWiseConv1d(in_channels=channels,
                                    out_channels=width,
                                    kernel_size=kernel_size,
                                    causal=False,
                                    block_size=1)
        self.residuals = []
        for _, layer in product(range(n_blocks), range(n_layers)):
            self.residuals.append(nn.Sequential(
                nn.ReLU(),
                BlockWiseConv1d(in_channels=width,
                                out_channels=width,
                                kernel_size=kernel_size,
                                causal=False,
                                block_size=2 ** layer),
                nn.ReLU(),
                BlockWiseConv1d(in_channels=width,
                                out_channels=width,
                                kernel_size=1,
                                block_size=1)
            ))
        self.residuals = nn.ModuleList(self.residuals)

        # Bottleneck
        self.final = nn.Sequential(
            BlockWiseConv1d(in_channels=width,
                            out_channels=bottleneck_dims,
                            kernel_size=1,
                            block_size=1),
            nn.AvgPool1d(kernel_size=hop_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.init(x)
        for residual in self.residuals:
            embedding = embedding + residual(embedding)
        embedding = self.final(embedding)
        return embedding
