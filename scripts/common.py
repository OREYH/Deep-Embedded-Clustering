"""Common loss and utility functions implemented with PyTorch."""

import torch
import torch.nn.functional as F


def sse(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Sum of squared errors along the feature dimension."""
    return torch.sum((true - pred) ** 2, dim=1)


def cce(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Sparse categorical cross entropy with logits."""
    return F.cross_entropy(pred, true.long(), reduction="none")


def bce(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy with logits summed over features."""
    return torch.sum(F.binary_cross_entropy_with_logits(pred, true, reduction="none"), dim=1)


def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Gaussian kernel matrix used for MMD."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
    kernel_input = torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2)) / float(dim)
    return kernel_input


def mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Maximum mean discrepancy between two samples."""
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()


def sampling(args) -> torch.Tensor:
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    z_mean, z_log_var = args
    epsilon = torch.randn_like(z_mean)
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon


def kl_regu(z_mean: torch.Tensor, z_log_sigma: torch.Tensor) -> torch.Tensor:
    """KL divergence regularizer for a normal prior."""
    kl_loss = 1 + z_log_sigma - z_mean.pow(2) - z_log_sigma.exp()
    kl_loss = kl_loss.sum(dim=-1)
    kl_loss *= -0.5
    return kl_loss
