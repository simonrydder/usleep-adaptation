import torch
import torch.nn as nn


class Conv1dAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gamma: int,
        kernel: int,
        activation_function: type[nn.Module],
        **conv_kwargs,
    ) -> None:
        super().__init__()

        hidden_dim = max(in_channels // gamma, 1)

        # Depth-wise Convolution
        self.W_down = nn.Conv1d(
            in_channels, hidden_dim, kernel_size=kernel, **conv_kwargs
        )

        # Activation function
        self.activation = activation_function()

        # Point-wise Convolution: kernel_size = 1
        self.W_up = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

        # Alpha: (N, C_out, L_out), N = L_out = 1 -> same scaling on all N and L?
        self.alpha_scale = nn.Parameter(
            torch.ones((1, out_channels, 1)),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.W_down(x)
        h = self.activation(x)
        h = self.W_up(x)

        y = h * self.alpha_scale  # Seems redundant
        return y
