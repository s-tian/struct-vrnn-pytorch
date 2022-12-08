import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):

    def __init__(self, num_steps, num_videos, split="train"):
        self.num_steps = num_steps
        self.num_videos = num_videos
        self.split = split
        self.a_dim = 4

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        return torch.rand(self.num_steps, 3, 64, 64), torch.rand(self.num_steps, self.a_dim)