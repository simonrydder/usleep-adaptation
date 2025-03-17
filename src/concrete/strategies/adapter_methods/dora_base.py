import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class DoRALayer:
    def __init__(
        self,
        r: int,
        dora_alpha: int,
        dora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.dora_alpha = dora_alpha
        # Optional dropout
        if dora_dropout > 0.0:
            self.dora_dropout = nn.Dropout(p=dora_dropout)
        else:
            self.dora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, DoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        dora_alpha: int = 1,
        dora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DoRALayer.__init__(
            self,
            r=r,
            dora_alpha=dora_alpha,
            dora_dropout=dora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            # Direction components (A and B)
            self.dora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.dora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # Magnitude component
            self.dora_magnitude = nn.Parameter(torch.ones(out_features))
            self.scaling = self.dora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if hasattr(self, "dora_A"):
            nn.init.kaiming_uniform_(self.dora_A, a=math.sqrt(5))
            nn.init.zeros_(self.dora_B)
            nn.init.ones_(self.dora_magnitude)

    def train(self, mode: bool = True) -> Module:
        def weight_transposed(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    delta_w = (
                        weight_transposed(self.dora_B @ self.dora_A) * self.scaling
                    )
                    self.weight.data -= delta_w * self.dora_magnitude.unsqueeze(1)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    delta_w = (
                        weight_transposed(self.dora_B @ self.dora_A) * self.scaling
                    )
                    self.weight.data += delta_w * self.dora_magnitude.unsqueeze(1)
                self.merged = True
        return self

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        def transpose(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            base_weight = transpose(self.weight)
            delta_w = (self.dora_B @ self.dora_A) * self.scaling
            weight_update = delta_w * self.dora_magnitude.unsqueeze(1)
            result = F.linear(input, base_weight + weight_update, bias=self.bias)
            return self.dora_dropout(result)
        else:
            return F.linear(input, transpose(self.weight), bias=self.bias)


class ConvDoRA(nn.Module, DoRALayer):
    def __init__(
        self,
        conv_module,
        in_channels,
        out_channels,
        kernel_size,
        r=0,
        dora_alpha=1,
        dora_dropout=0.0,
        merge_weights=True,
        **kwargs,
    ):
        super(ConvDoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)
        DoRALayer.__init__(
            self,
            r=r,
            dora_alpha=dora_alpha,
            dora_dropout=dora_dropout,
            merge_weights=merge_weights,
        )
        assert isinstance(kernel_size, int)
        if r > 0:
            self.dora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.dora_B = nn.Parameter(
                self.conv.weight.new_zeros(
                    (out_channels // self.conv.groups * kernel_size, r * kernel_size)
                )
            )
            self.dora_magnitude = nn.Parameter(torch.ones(out_channels))
            self.scaling = self.dora_alpha / self.r
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, "dora_A"):
            nn.init.kaiming_uniform_(self.dora_A, a=math.sqrt(5))
            nn.init.zeros_(self.dora_B)
            nn.init.ones_(self.dora_magnitude)

    def train(self, mode=True):
        super(ConvDoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    delta_w = (self.dora_B @ self.dora_A).view(
                        self.conv.weight.shape
                    ) * self.scaling
                    self.conv.weight.data -= delta_w * self.dora_magnitude.view(
                        -1, 1, 1, 1
                    )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    delta_w = (self.dora_B @ self.dora_A).view(
                        self.conv.weight.shape
                    ) * self.scaling
                    self.conv.weight.data += delta_w * self.dora_magnitude.view(
                        -1, 1, 1, 1
                    )
                self.merged = True

    def forward(self, input):
        if self.r > 0 and not self.merged:
            delta_w = (self.dora_B @ self.dora_A).view(
                self.conv.weight.shape
            ) * self.scaling
            weight_update = delta_w * self.dora_magnitude.view(-1, 1, 1, 1)
            return self.dora_dropout(
                self.conv._conv_forward(
                    input,
                    self.conv.weight + weight_update,
                    self.conv.bias,
                )
            )
        return self.conv(input)


class Conv2d(ConvDoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvDoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)
