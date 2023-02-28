from torch import Tensor
from utils import inner_product

import numpy as np
import torch

class Manifold:

    def __init__(self) -> None:
        pass

    def distance(self, p: Tensor, q: Tensor, keepdim: bool = False) -> Tensor:
        """
        Get the distance between two points `p` and `q` on the manifold.
        """
        pass

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Compute the exponential map, moving along the geodesic from point `p`
        in direction `v` in the tangent space of `p`.
        """
        pass

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Compute the logarithmic map, finding the tangent vector `v` at point `p`
        that points in the direction of the geodesic connecting `p` and `q`.
        """
        pass

    def proj(self, p: Tensor) -> Tensor:
        """
        Project a tensor `p` onto the manifold.
        """
        pass

    def proj_tangent(x: Tensor, p: Tensor) -> Tensor:
        """
        Project a tensor `x` onto the tangent space of `p`.
        """
        pass


class UnitSphere(Manifold):

    def __init__(self, dim: int, eps: float = 1e-8) -> None:

        super().__init__()

        self.dim = dim
        self.eps = eps

    def distance(self, p: Tensor, q: Tensor, keepdim: bool = False) -> Tensor:
        ip = inner_product(p, q, keepdim=keepdim)
        return torch.arccos(torch.clamp(ip, -1.0 + self.eps, 1.0 - self.eps))
    
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Compute the exponential map, moving along the geodesic from point `p`
        in direction `v` in the tangent space of `p`.
        """
        nv = torch.norm(v, dim=-1, keepdim=True)
        return torch.cos(nv)*p + torch.sin(nv)*v/nv

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Compute the logarithmic map, finding the tangent vector `v` at point `p`
        that points in the direction of the geodesic connecting `p` and `q`.
        """
        if torch.allclose(p, -q):

            # Create a direction vector in `self.dim`-1 dimensional space
            v = torch.zeros_like(p)
            v[...,0] = 1
            
            # Project onto tangent space of `p`
            v = self.proj_tangent(v, p)
            
            # Normalize to be `pi` length (equivalent to a step halfway around sphere)
            v = np.pi * v/torch.norm(v, dim=-1, keepdim=True)

            return v

        else:
            d = self.distance(p, q, keepdim=True)
            proj = self.proj_tangent(q, p)
            return d * proj/torch.norm(proj, dim=-1, keepdim=True)

    def proj(self, p: Tensor) -> Tensor:
        """
        Project a tensor `p` onto the manifold.
        """
        return p/torch.norm(p, dim=-1, keepdim=True)

    def proj_tangent(self, x: Tensor, p: Tensor) -> Tensor:
        """
        Project a tensor `x` onto the tangent space of `p`.
        """

        # Project onto the tangent plane at `p`
        y = x - inner_product(x, p, keepdim=True)/torch.norm(p, dim=-1, keepdim=True)*p
        return y