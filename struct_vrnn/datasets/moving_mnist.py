import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from hydra.utils import get_original_cwd


class MovingMNIST(Dataset):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, num_steps=20, num_digits=2, image_size=64, deterministic=True, split="train"):
        path = os.path.join(get_original_cwd(), "data")
        self.seq_len = num_steps
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False  # multi threaded loading
        self.channels = 3

        self.data = datasets.MNIST(
            path,
            train=(split == "train"),
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)
        # self.N = 100

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                     dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size - digit_size)
            sy = np.random.randint(image_size - digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    dy = -dy

                elif sy >= image_size - 32:
                    sy = image_size - 32 - 1
                    dy = -dy
                if sx < 0:
                    sx = 0
                    dx = -dx

                elif sx >= image_size - 32:
                    sx = image_size - 32 - 1
                    dx = -dx

                x[t, sy:sy + 32, sx:sx + 32, 0] += digit.numpy().squeeze()
                x[t, sy:sy + 32, sx:sx + 32, 1] += digit.numpy().squeeze()
                x[t, sy:sy + 32, sx:sx + 32, 2] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x > 1] = 1.
        # convert X from HWC to CHW
        x = np.transpose(x, (0, 3, 1, 2))
        return x, torch.zeros(self.seq_len, 4)
