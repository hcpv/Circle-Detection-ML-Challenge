from typing import Optional

import torch
from torch.utils.data import Dataset

from utils import *


class CircleDataset(Dataset):
    """
    Creates a custom Torch Dataset.
    Prepares n_samples number of data points.
    """

    def __init__(
        self,
        n_samples: int,
        noise_level: float = 0.5,
        img_size: int = 100,
        min_radius: Optional[int] = None,
        max_radius: Optional[int] = None,
    ):
        self.data = []
        if not min_radius:
            min_radius = img_size // 10
        if not max_radius:
            max_radius = img_size // 2
        assert max_radius > min_radius, "max_radius must be greater than min_radius"
        assert img_size > max_radius, "size should be greater than max_radius"
        assert noise_level >= 0, "noise should be non-negative"

        for i in range(n_samples):
            img, params = noisy_circle(
                img_size=img_size,
                min_radius=min_radius,
                max_radius=max_radius,
                noise_level=noise_level,
            )
            self.data.append([img, params])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, params = self.data[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        params_tensor = torch.tensor(params, dtype=torch.float32)
        return img_tensor, params_tensor
