import os

import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision


class HazePairDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True):
        super().__init__()
        self.dir = dir
        self.hazy_names = sorted(os.listdir(os.path.join(dir, "LQ")))
        self.clear_names = sorted(os.listdir(os.path.join(dir, "GT")))

        self.hazy_images = tuple(
            self._load_image(os.path.join(dir, "LQ", name)) for name in self.hazy_names
        )
        self.clear_images = tuple(
            self._load_image(os.path.join(dir, "GT", name)) for name in self.clear_names
        )

        self._validate_pairs()

        self.patch_size = np.int32(patch_size)
        self.n = np.int32(n)
        self.parse_patches = parse_patches
        self.transforms = transforms

    @staticmethod
    def _load_image(path):
        with open(path, "rb") as f:
            return PIL.Image.open(f).convert("RGB")

    def _validate_pairs(self):
        if len(self.hazy_names) != len(self.clear_names):
            raise ValueError("hazy and clear sets have different numbers of images")

        for hazy, clear in zip(self.hazy_names, self.clear_names):
            if not hazy.startswith(clear.split("_")[0]):
                raise ValueError(f"pair mismatch: {hazy} vs {clear}")

    @staticmethod
    def random_patch_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), th, tw

        top = np.random.randint(0, h - th, size=n)
        left = np.random.randint(0, w - tw, size=n)
        return top, left, th, tw

    def __getitem__(self, index):
        hazy_img = self.hazy_images[index]
        clear_img = self.clear_images[index]

        if self.parse_patches:
            top, left, h, w = self.random_patch_params(
                hazy_img, (self.patch_size, self.patch_size), self.n
            )

            hazy_patches = [hazy_img.crop((left[k], top[k], left[k] + w, top[k] + h))
                            for k in range(self.n)]
            clear_patches = [clear_img.crop((left[k], top[k], left[k] + w, top[k] + h))
                             for k in range(self.n)]

            return torch.stack(
                [torch.cat([self.transforms(hazy), self.transforms(clear)], dim=0)
                 for hazy, clear in zip(hazy_patches, clear_patches)]
            ), os.path.splitext(self.hazy_names[index])[0]

        max_size = 3072

        wd_new, ht_new = hazy_img.size
        if ht_new > wd_new and ht_new > max_size:
            wd_new = int(np.ceil(wd_new * max_size / ht_new))
            ht_new = max_size
        elif ht_new <= wd_new and wd_new > max_size:
            ht_new = int(np.ceil(ht_new * max_size / wd_new))
            wd_new = max_size
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))

        hazy_img = hazy_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
        clear_img = clear_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)

        return torch.cat([self.transforms(hazy_img), self.transforms(clear_img)], dim=0), \
            self.hazy_names[index]

    def __len__(self):
        return len(self.hazy_names)
