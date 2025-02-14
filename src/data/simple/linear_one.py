import torch
from torch.utils.data import Dataset


class LinearOne(Dataset):
    def __init__(self, samples=1000) -> None:
        super().__init__()

        self.x = torch.rand(samples, 2)
        W = torch.tensor([2.0, -4.0]).view(1, -1)
        b = torch.tensor([5.0])
        self.y = self.x.matmul(W.T) + b
        self.y = self.y.view(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
