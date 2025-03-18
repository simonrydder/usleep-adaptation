import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding="same",
                padding_mode="zeros",
            ),
            activation,
            nn.BatchNorm1d(self.out_channels),
        )

        nn.init.xavier_uniform_(self.layer[1].weight)
        nn.init.zeros_(self.layer[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.conv_layer = ConvLayer(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            dilation=1,
            activation=nn.ReLU(),
        )

        self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layer(x)
        shortcut = x
        y = self.maxpool(x)
        return y, shortcut


class Decoder(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv_layer = ConvLayer(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
            dilation=1,
            activation=nn.ReLU(),
        )
        self.conv_layer_2 = ConvLayer(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            dilation=1,
            activation=nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv_layer(x)
        x = self._concatenation(x, shortcut)
        x = self.conv_layer_2(x)
        return x

    def _concatenation(self, x: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        diff = max(0, x.shape[2] - shortcut.shape[2])
        start = diff // 2 + diff % 2

        reshaped = x[:, :, start : start + shortcut.shape[2]]
        z = torch.cat((shortcut, reshaped), dim=1)
        return z


class SegmentClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class Usleep(nn.Module):
    def __init__(
        self,
        depth: int,
    ):
        super().__init__()

        self.encoders = nn.ModuleList([Encoder() for _ in range(depth)])
        self.bottom = ConvLayer()
        self.decoders = nn.ModuleList([Decoder() for _ in range(depth)])
        self.segment_classifier = SegmentClassifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcuts = []
        for layer in self.encoders:
            x, shortcut = layer(x)
            shortcuts.append(shortcut)

        x = self.bottom(x)

        for layer, shortcut in zip(self.decoders, reversed(shortcuts)):
            x = layer(x, shortcut)

        x = self.segment_classifier(x)

        return x
