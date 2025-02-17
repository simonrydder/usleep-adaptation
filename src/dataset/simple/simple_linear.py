from typing import Literal

import torch
from torch.utils.data import Dataset


class SimpleLinear(Dataset):
    def __init__(self, num_samples=1000, distribution: Literal[1, 2] = 1) -> None:
        super().__init__()

        self.num_samples = num_samples

        match distribution:
            case 1:
                self.W = torch.tensor([2.0, -4.0]).view(1, -1)
                self.b = torch.tensor([5.0])
            case 2:
                self.W = torch.tensor([2.0, -3.0]).view(1, -1)
                self.b = torch.tensor([7.0])
            case _:
                raise NotImplementedError()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(2)

        y = x.matmul(self.W.T) + self.b
        y = y.view(-1, 1)

        return x, y
