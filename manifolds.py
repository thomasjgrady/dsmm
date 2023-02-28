from torch import Tensor

import torch
import warnings

class Manifold:
    """
    Base Riemannian manifold object. All subclasses must implement
        - Distance function
        - Exponential map
        - Logarithmic map
        - Projection of arbitrary vector onto manifold
        - Projection of arbitrary vector onto tangent space of point on manifold
        - Generation of an embedding for values of a discrete sequence.

    Optionally, subclasses may implement
        - Visualization

    Manifold operations assume that the last dimension of each argument tensor
    corresponds to the embedding dimension of the manifold.
    """

    def __init__(self) -> None:
        pass

    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Computes the distance between `p` and `q` on the manifold.
        """
        pass

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Computes the exponential map. I.e., moves along the geodesic at `p`
        with direction and magnitude `v`.
        """
        pass

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Computes the logarithmic map. I.e. computes the vector `v` in the
        tangent space of `p` s.t. `exp(p, v) = q`.
        """
        pass

    def proj(self, x: Tensor) -> Tensor:
        """
        Projects an arbitrary vector `x` onto the manifold.
        """
        pass

    def proj_tangent(self, x: Tensor, p: Tensor) -> Tensor:
        """
        Projects and arbitrary vector `x` onto the tangent space of `p`.
        """
        pass

    def generate_embedding(self, n: int) -> Tensor:
        """
        Generates an embedding for `n` points in the manifold.
        """
        pass

    def draw(self, ax, *args, **kwargs):
        """
        Draws onto the axis `ax` with optional arguments.
        """
        warnings.warn(f'Attempted to draw object {self} which does not implement `draw`')
        pass

class UnitSphere(Manifold):
    """
    Manifold of the L2 unit sphere in an arbitrary number of dimensions.
    """

    def __init__(self, dim: int = 3, eps: float = 1e-6) -> None:

        super().__init__()

        self.dim = dim
        self.eps = eps

    def distance(self, p: Tensor, q: Tensor, keepdim: bool = False) -> Tensor:
        p_bc, q_bc = torch.broadcast_tensors(p, q)
        ip = torch.einsum('...d,...d->...', p, q)
        d = torch.arccos(torch.clamp(ip, -1.0 + self.eps, 1.0 - self.eps))
        if keepdim:
            d = d.unsqueeze(-1)
        return d

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        p_bc, v_bc = torch.broadcast_tensors(p, v)
        v_bc_norm = torch.norm(v_bc, dim=-1, keepdim=True)
        return torch.cos(v_bc_norm)*p_bc + torch.sin(v_bc_norm)*v_bc/v_bc_norm

    def _log_opposite(self, p: Tensor, q: Tensor) -> Tensor:
        v = torch.zeros_like(p)
        v[...,0] = 1
        v = self.proj_tangent(v, p)
        v = np.pi * v/torch.norm(v, dim=-1, keepdim=True)
        return v

    def _log_normal(self, p: Tensor, q: Tensor) -> Tensor:
        d = self.distance(p, q, keepdim=True)
        proj = self.proj_tangent(p, q)
        return d*proj/torch.norm(proj, dim=-1, keepdim=True)

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        p_bc, q_bc = torch.broadcast_tensors(p, q)
        out = torch.where(p_bc == -q_bc, _log_opposite(p_bc, q_bc), _log_normal(p_bc, q_bc))
        return out

    def proj(self, x: Tensor) -> Tensor:
        return x/torch.norm(x, dim=-1, keepdim=True)

    def proj_tangent(self, p: Tensor, x: Tensor) -> Tensor:
        p_bc, x_bc = torch.broadcast_tensors(p, x)
        ip = torch.einsum('...d,...d->...', p_bc, x_bc).unsqueeze(-1)
        return x_bc - ip*p_bc

    def generate_embedding(self, n: int) -> Tensor:

        # For now, only implement fibonacci lattice in 3-space.
        if self.dim == 3:

            pts = torch.arange(0, n).to(torch.float32)
            pts += 0.5

            phi = torch.acos(1.0 - 2.0*pts/n)
            theta = np.pi * (1 + np.sqrt(5)) * pts

            xs = torch.cos(theta) * torch.sin(phi)
            ys = torch.sin(theta) * torch.sin(phi)
            zs = torch.cos(phi)

            out = torch.cat((
                xs.view(n, 1),
                ys.view(n, 1),
                zs.view(n, 1)
            ), dim=-1)

            return out

        else:
            raise Exception('unimplemented')

    def draw(self, ax, *args, **kwargs):
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        return ax.plot_wireframe(x, y, z, *args, **kwargs)
