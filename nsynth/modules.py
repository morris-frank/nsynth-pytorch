from typing import Tuple, Optional

import torch
from torch import dtype as torch_dtype
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
                 kernel_size: int = 1,
                 block_size: int = 1,
                 causal: bool = False,
                 **kwargs):
        """
        Block-wise 1D-Convolution as used in original NSynth
        [http://arxiv.org/abs/1704.01279].
        Args:
            in_channels: Num of Channels of the Input
            out_channels: Num of Filters in the Convolution
            kernel_size: Size of the Filters
            block_size: Length of the Blocks we split the input into,
                with block size == 1 ⇒ same as nn.Conv1d!
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


class DilatedQueue:
    def __init__(self, size: int, data: Optional[torch.Tensor] = None,
                 channels: int = 1, dilation: int = 1,
                 dtype: torch_dtype = torch.float32):
        self.idx_en, self.idx_de = 0, 0
        self.channels, self.size = channels, size
        self.dtype = dtype
        self.dilation = dilation

        self.data = data
        if data is None:
            self.reset()

    def enqueue(self, x):
        self.data[:, self.idx_en] = x
        self.idx_en = (self.idx_en + 1) % self.size

    def dequeue(self, num_deq=1):
        start = self.idx_de - ((num_deq - 1) * self.dilation)
        end = self.idx_de
        if start < 0:
            t1 = self.data[:, start::self.dilation]
            t2 = self.data[:, self.idx_de % self.dilation:end:self.dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:end:self.dilation]

        self.idx_de = (self.idx_de + 1) % self.size
        return t

    def reset(self, device: str = 'cpu'):
        self.idx_en, self.idx_de = 0, 0
        self.data = torch.zeros((self.channels, self.size),
                                dtype=self.dtype).to(device)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Module()
        self.decoder = nn.Module()

    @staticmethod
    def loss_function(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
