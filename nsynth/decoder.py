from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from .functional import shift1d
from .modules import BlockWiseConv1d


class WaveNetDecoder(nn.Module):
    """
    WaveNet as described NSynth [http://arxiv.org/abs/1704.01279].
    """

    def __init__(self,
                 n_layers: int = 10,
                 n_blocks: int = 3,
                 width: int = 512,
                 skip_width: int = 256,
                 channels: int = 1,
                 quantization_channels: int = 256,
                 bottleneck_dims: int = 16,
                 kernel_size: int = 3):
        """
        WaveNet as described NSynth [http://arxiv.org/abs/1704.01279].

        This WaveNet has some differences to the original WaveNet. Namely:
        · It uses a conditioning on all layers, input always the same
          conditioning, added to the dilated values (features and gates) as well
          as after  the final skip convolution.
        · The skip connection does not start at 0 but comes from a 1×1
          Convolution from the initial Convolution.

        Args:
            n_layers: Number of layers in each block
            n_blocks: Number of blocks
            width: The width/size of the hidden layers
            skip_width: The width/size of the skip connections
            channels: Number of input channels
            quantization_channels: Number of final output channels
            bottleneck_dims: Dim/width/size of the conditioning, output of the
                encoder
            kernel_size: Kernel-size to use
        """
        super(WaveNetDecoder, self).__init__()
        self.width = width
        self.n_stages, self.n_layers = n_blocks, n_layers

        self.initial_dilation = BlockWiseConv1d(in_channels=channels,
                                                out_channels=width,
                                                kernel_size=kernel_size,
                                                block_size=1,
                                                causal=True)
        self.initial_skip = nn.Conv1d(channels, skip_width, 1)

        self.dilations = self._make_conv_list(width, 2 * width, kernel_size)
        self.conds = self._make_conv_list(bottleneck_dims, 2 * width, 1)
        self.residuals = self._make_conv_list(width, width, 1)
        self.skips = self._make_conv_list(width, skip_width, 1)

        self.final_skip = nn.Sequential(nn.ReLU(),
                                        nn.Conv1d(skip_width, skip_width, 1))
        self.final_cond = nn.Conv1d(bottleneck_dims, skip_width, 1)
        self.final_quant = nn.Sequential(nn.ReLU(),
                                         nn.Conv1d(skip_width,
                                                   quantization_channels, 1))

    def _make_conv_list(self, in_channels: int, out_channels: int,
                        kernel_size: int) -> nn.ModuleList:
        """
        A little helper function for generating lists of Convolutions. Will give
        list of n_blocks × n_layers number of convolutions. If kernel_size is
        bigger than one we use the BlockWise Convolution and calculate the block
        size from the power-2 dilation otherwise we always use the same
        1×1-conv1d.

        Args:
            in_channels: In channels
            out_channels: out channels
            kernel_size: kernel size

        Returns:
        ModuleList of len self.n_blocks * self.n_layers
        """
        conv = nn.Conv1d if kernel_size == 1 else BlockWiseConv1d
        module_list = []
        args = (in_channels, out_channels, kernel_size)
        for _, layer in product(range(self.n_stages), range(self.n_layers)):
            opt = () if kernel_size == 1 else (2 ** layer, True)
            module_list.append(conv(*(args + opt)))
        return nn.ModuleList(module_list)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = shift1d(x, -1)
        x = self.initial_dilation(x)
        skip = self.initial_skip(x)

        layers = (self.dilations, self.conds, self.residuals, self.skips)
        for l_dilation, l_cond, l_residual, l_skip in zip(*layers):
            dilated = l_dilation(x)
            dilated += l_cond(embedding)
            filters = F.sigmoid(dilated[:, :self.width, :])
            gates = F.tanh(dilated[:, self.width:, :])
            pre_res = filters * gates

            x += l_residual(pre_res)
            skip += l_skip(pre_res)

        skip = self.final_skip(skip)
        skip += self.final_cond(embedding)
        quant_skip = self.final_quant(skip)
        return quant_skip
