from dataclasses import dataclass, field
from manifolds import UnitSphere
from torch import Tensor
from typing import *

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fibonacci_spherical_lattice(n_points: int, dim: int = 3) -> Tensor:

    assert dim == 3, 'dimensions != 3 are unimplemented!'

    pts = torch.arange(0, n_points).to(torch.float32)
    pts += 0.5

    phi = torch.acos(1.0 - 2.0*pts/n_points)
    theta = np.pi * (1 + np.sqrt(5)) * pts

    xs = torch.cos(theta) * torch.sin(phi)
    ys = torch.sin(theta) * torch.sin(phi)
    zs = torch.cos(phi)

    out = torch.cat((
        xs.view(n_points, 1),
        ys.view(n_points, 1),
        zs.view(n_points, 1)
    ), dim=-1)

    return out

@dataclass
class ModelConfig:

    # Model vocabulary 
    vocab_size: int = 50_257

    # Context width
    n_tokens: int = 64

    # Manifold dimensionality
    n_embed: int = 3

    # Number of attention heads
    n_heads: int = 1

    # Hidden size in MLP
    n_hidden: int = 0

    # Number of attention-mlp blocks
    n_blocks: int = 4

    # Hidden activation function
    act: Callable = F.gelu

    # Path to sphere embedding (will be generated using Fibonacci lattice)
    # if none provided
    word_embedding_path: Optional[str] = None

    # Model device
    device: torch.device = torch.device('cuda')

    # Model datatype
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.word_embedding_path is None:
            self.word_embedding_path = f'{self.vocab_size}_{self.n_embed}_embeddings.pt'
        
        if not os.path.exists(self.word_embedding_path):
            print('Generating word embeddings... ', end='', flush=True)
            embeddings = fibonacci_spherical_lattice(self.vocab_size, self.n_embed) \
                .to(device=self.device, dtype=self.dtype)
            torch.save(embeddings, self.word_embedding_path)
            print('done.', flush=True)
        
        if self.n_hidden == 0:
            self.n_hidden = self.n_embed*4

class CausalAttention(nn.Module):
    """
    Performs attention with causal masking on the given input sequence.
    The calculation is agnostic to sequence length.
    """

    def __init__(self, config: ModelConfig) -> None:

        super().__init__()

        self.config = config

        self.lift0 = nn.Linear(config.n_embed, config.n_embed*2, device=config.device, dtype=config.dtype)
        self.lift1 = nn.Linear(config.n_embed, config.n_embed,   device=config.device, dtype=config.dtype)
        self.proj  = nn.Linear(config.n_embed, config.n_embed,   device=config.device, dtype=config.dtype)
        self.mask = torch.tril(torch.ones(config.n_tokens, config.n_tokens, device=config.device, dtype=torch.long)) \
                .view(1, 1, config.n_tokens, config.n_tokens)

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:

        n_batch, n_tokens, n_embed = x0.shape
        n_heads = self.config.n_heads

        q, k = self.lift0(x0).split(n_embed, dim=-1)
        v    = self.lift1(x1)

        q = q.view(n_batch, n_tokens, n_heads, n_embed // n_heads).transpose(1, 2)
        k = k.view(n_batch, n_tokens, n_heads, n_embed // n_heads).transpose(1, 2)
        v = v.view(n_batch, n_tokens, n_heads, n_embed // n_heads).transpose(1, 2)
        
        mask = self.mask[:,:,:n_tokens,:n_tokens]
        a = (q @ k.transpose(-1, -2))/np.sqrt(n_embed // n_heads)
        a.masked_fill_(mask == 0, float('-inf'))
        a = F.softmax(a, dim=-1)
        
        y = (a @ v).transpose(1, 2).contiguous().view(n_batch, n_tokens, n_embed)
        y = self.proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config: ModelConfig) -> None:

        super().__init__()

        self.config = config

        self.w0 = nn.Linear(config.n_embed, config.n_hidden, device=config.device, dtype=config.dtype)
        self.w1 = nn.Linear(config.n_hidden, config.n_embed, device=config.device, dtype=config.dtype)
        self.act = config.act

    def forward(self, x: Tensor) -> Tensor:
        x = self.w0(x)
        x = self.act(x)
        x = self.w1(x)
        return x

class AttentionBlock(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        
        super().__init__()

        self.config = config
        self.attn = CausalAttention(config)
        self.mlp  = MLP(config)
        self.ln0  = nn.LayerNorm(config.n_embed, device=config.device, dtype=config.dtype)
        self.ln1  = nn.LayerNorm(config.n_embed, device=config.device, dtype=config.dtype)

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        x = x0 + self.attn(x0, x1)
        x = self.ln0(x)
        x = x + self.mlp(x)
        x = self.ln1(x)
        return x

class Model(nn.Module):

    def __init__(self, config: ModelConfig) -> None:

        super().__init__()

        self.config = config
        self.manifold = UnitSphere(config.n_embed)
        self.word_vecs = torch.load(config.word_embedding_path)

        self.blocks = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_blocks)])

    def get_word_vecs(self, tokens: Tensor) -> Tensor:
        n_batch, n_tokens = tokens.shape
        return torch.index_select(self.word_vecs, dim=0, index=tokens.flatten()).reshape(n_batch, n_tokens, self.config.n_embed)

    def forward(self, x: Tensor, word_vecs: Tensor, t: float) -> Tensor:
        
        # Get shape info
        n_batch, _, n_embed = x.shape
        _, n_vecs, _ = word_vecs.shape

        # Compute tangents from each input point to the current point
        x_in = x
        x = x.repeat(1, n_vecs, 1)
        tangents = self.manifold.log(word_vecs, x)

        # Perform attention weighting by the value of t based on how close we are
        # to the solution
        x0 = t*x + (1-t)*tangents
        for b in self.blocks:
            x0 = b(x0, tangents)

        # Average output, giving (n_batch, n_embed)
        tangent = torch.mean(x0, dim=1, keepdim=True)

        # Project onto tangent space of x
        tangent = self.manifold.proj_tangent(tangent, x_in)

        return tangent