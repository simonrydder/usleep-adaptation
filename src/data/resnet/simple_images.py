import torch
from torch.utils.data import Dataset


class SimpleImages(Dataset):
    def __init__(
        self, num_samples, num_classes, image_size=(3, 224, 224), distribution="normal"
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.distribution = distribution

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Choose the distribution for generating images.
        if self.distribution == "normal":
            # Standard normal distribution: mean 0, std 1.
            image = torch.randn(*self.image_size)
        elif self.distribution == "shifted":
            # Shifted normal: mean 3, std 2 (a noticeably different distribution).
            image = torch.randn(*self.image_size) * 2 + 3
        elif self.distribution == "uniform":
            # Uniform distribution between 0 and 1.
            image = torch.rand(*self.image_size)
        else:
            # Fallback to standard normal.
            image = torch.randn(*self.image_size)

        # Random label from 0 to num_classes-1.
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label
