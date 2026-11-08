import os

import torch
import torchvision.utils as tvu


def save_image(img, file_path, **kwargs):
    save_dir = os.path.dirname(file_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    tvu.save_image(img, file_path, **kwargs)


def save_checkpoint(state, filename):
    save_dir = os.path.dirname(filename)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, filename + ".pth.tar")


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    return torch.load(path, map_location=device)
