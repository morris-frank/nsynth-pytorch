from itertools import product
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from .modules import BlockWiseConv1d, DilatedQueue


class WaveNetDecoder(nn.Module):
    """
    WaveNet as described NSynth [http://arxiv.org/abs/1704.01279].

    This WaveNet has some differences to the original WaveNet. Namely:
    · It uses a conditioning on all layers, input always the same
      conditioning, added to the dilated values (features and gates) as well
      as after  the final skip convolution.
    · The skip connection does not start at 0 but comes from a 1×1
      Convolution from the initial Convolution.
    """

    def __init__(self,
                 n_layers: int = 10,
                 n_blocks: int = 3,
                 width: int = 512,
                 skip_width: int = 256,
                 channels: int = 1,
                 quantization_channels: int = 256,
                 bottleneck_dims: int = 16,
                 kernel_size: int = 3,
                 gen: bool = False):
        """
        :param n_layers: Number of layers in each block
        :param n_blocks: Number of blocks
        :param width: The width/size of the hidden layers
        :param skip_width: The width/size of the skip connections
        :param channels: Number of input channels
        :param quantization_channels: Number of final output channels
        :param bottleneck_dims: Dim/width/size of the conditioning, output
            of the encoder
        :param kernel_size: Kernel-size to use
        :param gen: Is this generation ?
        """
        super(WaveNetDecoder, self).__init__()
        self.width = width
        self.n_stages, self.n_layers = n_blocks, n_layers
        # The compound dilation (input to last layer in each block):
        self.scale_factor = 2 ** (n_layers - 1)
        self.receptive_field = 2 ** n_layers * n_blocks
        self.quantization_channels = quantization_channels
        self.gen = gen
        self.kernel_size = kernel_size

        self.initial_dilation = BlockWiseConv1d(in_channels=channels,
                                                out_channels=width,
                                                kernel_size=kernel_size,
                                                block_size=1,
                                                causal=True)

        self.initial_skip = BlockWiseConv1d(width, skip_width, 1)

        self.dilations = self._make_conv_list(width, 2 * width, kernel_size,
                                              not gen)
        self.conds = self._make_conv_list(bottleneck_dims, 2 * width, 1, False)
        self.residuals = self._make_conv_list(width, width, 1, False)
        self.skips = self._make_conv_list(width, skip_width, 1, False)

        self.queues = []
        for _, l in product(range(self.n_stages), range(self.n_layers)):
            self.queues.append(
                DilatedQueue(size=(kernel_size - 1) * 2 ** l + 1,
                             channels=width, dilation=2 ** l)
            )

        self.upsampler = nn.Upsample(scale_factor=self.scale_factor,
                                     mode='nearest')

        self.final_skip = nn.Sequential(
            nn.ReLU(),
            BlockWiseConv1d(skip_width, skip_width, 1)
        )

        self.final_cond = BlockWiseConv1d(bottleneck_dims, skip_width, 1)

        self.final_quant = nn.Sequential(
            nn.ReLU(),
            BlockWiseConv1d(skip_width, quantization_channels, 1)
        )

    def _make_conv_list(self, in_channels: int, out_channels: int,
                        kernel_size: int, dilate: bool) -> nn.ModuleList:
        """
        A little helper function for generating lists of Convolutions. Will
        give list of n_blocks × n_layers number of convolutions. If kernel_size
        is bigger than one we use the BlockWise Convolution and calculate the
        block size from the power-2 dilation otherwise we always use the same
        1×1-conv1d.

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param dilate: Whether to dilate in each step
        :return: ModuleList of len self.n_blocks * self.n_layers
        """
        module_list = []
        for _, layer in product(range(self.n_stages), range(self.n_layers)):
            block_size = 2 ** layer if dilate else 1
            module_list.append(BlockWiseConv1d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               block_size=block_size,
                                               causal=kernel_size != 1))
        return nn.ModuleList(module_list)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor,
                conditionals: Optional[List[torch.Tensor]] = None) \
            -> torch.Tensor:
        """

        :param x:
        :param embedding:
        :param conditionals: (Optional) contains list of all upsampled
            conditionals. Used for generation. If given do not give an
            embedding.
        :return:
        """
        x = self.initial_dilation(x)
        skip = self.initial_skip(x)

        conds = conditionals or self.conds

        layers = (self.dilations, conds, self.residuals, self.skips,
                  self.queues)
        for l_dilation, cond, l_residual, l_skip, queue in zip(*layers):
            if self.gen:
                queue.enqueue(x.squeeze())
                dilated = queue.dequeue(num_deq=self.kernel_size)
                dilated = dilated.unsqueeze(0)
            else:
                dilated = x
            dilated = l_dilation(dilated)
            if self.gen:
                dilated = dilated[:, :, 1].unsqueeze(-1)
            if conditionals:
                dilated = dilated + cond
            else:
                dilated = dilated + self.upsampler(cond(embedding))
            filters = torch.sigmoid(dilated[:, :self.width, :])
            gates = torch.tanh(dilated[:, self.width:, :])
            pre_res = filters * gates

            x = x + l_residual(pre_res)  # Is this correct????
            skip = skip + l_skip(pre_res)

        skip = self.final_skip(skip)
        if conditionals:
            skip = skip + conds[-1]
        else:
            skip = skip + self.upsampler(self.final_cond(embedding))
        quant_skip = self.final_quant(skip)
        return quant_skip

    def generate(self, x: torch.Tensor, conditionals, length: int, device: str,
                 temp: float = 1.):
        for queue in self.queues:
            queue.reset(device)

        rem_length = length - x.numel()

        # Fill queues with initial values
        for i in trange(x.numel() - 1):
            inp = x[0, 0, i:i + 1].view(1, 1, 1)
            _ = self(inp, None, conditionals)

        generation = torch.zeros(length)
        for i in trange(rem_length):
            logits = self(inp, None, conditionals).squeeze()

            if temp > 0:
                prob = F.softmax(logits / temp, dim=0)
                c = torch.multinomial(prob, 1).float()
            else:
                c = torch.argmax(logits).float()
            c = (c - 128.) / 128.
            generation[i] = c.cpu()
            inp = c.view(1, 1, 1)
        return generation
