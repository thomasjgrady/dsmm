from torch import Tensor

import torch

def inner_product(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:

    sx = x.shape
    sy = y.shape
    x = x.view(-1, sx[-1])
    y = y.view(-1, sy[-1])
    out = torch.einsum('bn,bn->b', x, y)

    if keepdim:
        return out.view(*sx[:-1], 1)
    else:
        return out.view(*sx[:-1])