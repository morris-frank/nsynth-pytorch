from typing import Tuple

import torch
from torch import nn

from .functional import time_to_batch, batch_to_time


class BlockWiseConv1d(nn.Conv1d):
    """
    Block-wise 1D-Convolution as used in original NSynth
    [http://arxiv.org/abs/1704.01279].
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 block_size: int,
                 causal: bool = False,
                 **kwargs):
        """
        Block-wise 1D-Convolution as used in original NSynth
        [http://arxiv.org/abs/1704.01279].
        Args:
            in_channels: Num of Channels of the Input
            out_channels: Num of Filters in the Convolution
            kernel_size: Size of the Filters
            block_size: Length of the Blocks we split the input into
            causal: Whether to do it Causal or not
            **kwargs:
        """
        super(BlockWiseConv1d, self).__init__(in_channels, out_channels,
                                              kernel_size, **kwargs)
        self.block_size = block_size

        assert kernel_size % 2 != 0
        assert block_size >= 1

        if causal:
            pad = (kernel_size - 1, 0)
        else:
            pad = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(pad, 0)

        self.weight_init()

    def weight_init(self):
        weight, bias = self.weight, self.bias
        self.weight = nn.init.xavier_uniform_(weight)
        self.bias = nn.init.constant_(bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = time_to_batch(x, self.block_size)
        y = self.pad(y)
        y = super(BlockWiseConv1d, self).forward(y)
        y = batch_to_time(y, self.block_size)
        return y


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Module()
        self.decoder = nn.Module()

    @staticmethod
    def loss_function(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
