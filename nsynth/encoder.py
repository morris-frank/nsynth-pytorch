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
                 hop_length: int = 512,
                 use_bias: bool = True):
        """
        :param channels: Number of input channels
        :param n_layers: Number of layers in each stage in the encoder
        :param n_blocks: Number of stages
        :param width: Size of the hidden channels in all layers
        :param kernel_size: KS for all 1D-convolutions
        :param bottleneck_dims: Final number of features
        :param hop_length: Final bottleneck pooling
        :param use_bias: Whether to use bias in all the convolutions.
        """
        super(TemporalEncoder, self).__init__()

        self.encoder = []
        self.encoder.append(
            BlockWiseConv1d(in_channels=channels,
                            out_channels=width,
                            kernel_size=kernel_size,
                            causal=False,
                            block_size=1,
                            bias=use_bias)
        )
        for _, layer in product(range(n_blocks), range(n_layers)):
            dilation = 2 ** layer
            self.encoder.extend([
                nn.ReLU(),
                BlockWiseConv1d(in_channels=width,
                                out_channels=width,
                                kernel_size=kernel_size,
                                causal=False,
                                block_size=dilation,
                                bias=use_bias),
                nn.ReLU(),
                nn.Conv1d(width, width, 1, bias=use_bias)
            ])

        # Bottleneck
        self.encoder.append(
            nn.Conv1d(in_channels=width,
                      out_channels=bottleneck_dims,
                      kernel_size=1,
                      bias=use_bias)
        )
        self.encoder.append(
            nn.AvgPool1d(kernel_size=hop_length)
        )
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
