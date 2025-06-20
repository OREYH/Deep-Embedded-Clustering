import torch
import torch.nn.functional as F


def sse(true, pred):
    return torch.sum((true - pred) ** 2, dim=1)


def cce(true, pred):
    return F.cross_entropy(pred, true.long(), reduction="none")


def bce(true, pred):
    return torch.sum(F.binary_cross_entropy_with_logits(pred, true, reduction='none'), dim=1)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
    kernel_input = torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2)) / float(dim)
    return kernel_input


def mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = torch.randn_like(z_mean)
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon


def kl_regu(z_mean, z_log_sigma):
    kl_loss = 1 + z_log_sigma - z_mean.pow(2) - torch.exp(z_log_sigma)
    kl_loss = torch.sum(kl_loss, dim=-1)
    kl_loss *= -0.5
    return kl_loss
