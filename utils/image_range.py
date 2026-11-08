import torch


def to_bipolar_range(x):
    return 2 * x - 1.0


def to_image_range(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
