import math
import torch
import torch.nn.functional as F


def time_to_batch(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Chops of a time-signal into a batch of equally-long signals.

    Args:
        x: input signal sized [Batch × Channels × Length]
        block_size: size of the blocks

    Returns:
        Tensor with size:
        [Batch * block size × Channels × Length/block_size]
    """
    assert x.ndimension() == 3
    nbatch, c, len = x.shape

    y = torch.reshape(x, [nbatch, c, len // block_size, block_size])
    y = y.permute(0, 3, 1, 2)
    y = torch.reshape(y, [nbatch * block_size, c, len // block_size])
    return y.contiguous()


def batch_to_time(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Inverse of time_to_batch. Concatenates a batched time-signal back to
    correct time-domain.

    Args:
        x: The batched input size [Batch * block_size × Channels × Length]
        block_size: size of the blocks used for encoding

    Returns:
        Tensor with size: [Batch × channels × Length * block_size]
    """
    assert x.ndimension() == 3
    batch_size, channels, k = x.shape
    y = torch.reshape(x, [batch_size // block_size, block_size, channels, k])
    y = y.permute(0, 2, 3, 1)
    y = torch.reshape(y, [batch_size // block_size, channels, k * block_size])
    return y.contiguous()


def shift1d(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shifts a Tensor to the left or right and pads with zeros.

    Args:
        x: Input Tensor [N×C×L]
        shift: the shift, negative for left shift, positive for right

    Returns:
        the shifted tensor, same size
    """
    assert x.ndimension() == 3
    length = x.shape[2]
    pad = [-min(shift, 0), max(shift, 0)]
    y = F.pad(x, pad)
    y = y[:, :, pad[1]:pad[1] + length]
    return y.contiguous()


def encode_μ_law(x: torch.Tensor, μ: int = 255, cast: bool = False)\
        -> torch.Tensor:
    """
    Encodes the input tensor element-wise with μ-law encoding

    Args:
        x: tensor
        μ: the size of the encoding (number of possible classes)
        cast: whether to cast to int8

    Returns:

    """
    out = torch.sign(x) * torch.log(1 + μ * torch.abs(x)) / math.log(1 + μ)
    out = torch.floor(out * math.ceil(μ / 2))
    if cast:
        out = out.type(torch.int8)
    return out


def decode_μ_law(x: torch.Tensor, μ: int = 255) -> torch.Tensor:
    """
    Applies the element-wise inverse μ-law encoding to the tensor.

    Args:
        x: input tensor
        μ: size of the encoding (number of possible classes)

    Returns:
        the decoded tensor
    """
    x = x.type(torch.float32)
    # out = (x + 0.5) * 2. / (μ + 1)
    out = x / math.ceil(μ / 2)
    out = torch.sign(out) / μ * (torch.pow(1 + μ, torch.abs(out)) - 1)
    return out
