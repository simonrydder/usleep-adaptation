import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class KronALayer:
    """
    Base class analogous to LoRALayer, storing key hyperparameters
    and a possible dropout. 'r' here dictates the Kronecker factor sizes.
    """

    def __init__(
        self,
        r: int,
        kron_alpha: int,
        kron_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.kron_alpha = kron_alpha
        # Optional dropout
        if kron_dropout > 0.0:
            self.kron_dropout = nn.Dropout(p=kron_dropout)
        else:
            # No-op
            self.kron_dropout = lambda x: x

        # Whether we have currently "merged" the KronA weights into the base weight
        self.merged = False
        self.merge_weights = merge_weights


class KronALinear(nn.Linear, KronALayer):
    """
    Example Kronecker-based adaptation in a Dense layer.

    We define two parameters kronA_A, kronA_B whose Kronecker product
    is shaped identically to the main layer's weight. The base weight
    is frozen, and these KronA parameters are learned to adapt the layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,  # 'r' controls Kron factor shapes
        kron_alpha: int = 1,
        kron_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # If the weight is transposed in the original layer
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        KronALayer.__init__(
            self,
            r=r,
            kron_alpha=kron_alpha,
            kron_dropout=kron_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out

        # If we do not want KronA (r=0), just behave like a normal Linear
        if r > 0:
            # Check that in_features, out_features are multiples of r
            if (out_features % r) != 0 or (in_features % r) != 0:
                raise ValueError(
                    f"For this simple KronA example, out_features ({out_features}) "
                    f"and in_features ({in_features}) must be multiples of r ({r})."
                )

            # Define Kronecker factors so that kron(kronA_A, kronA_B) has shape = (out_features, in_features).
            self.kronA_A = nn.Parameter(
                self.weight.new_zeros((out_features // r, in_features // r))
            )
            self.kronA_B = nn.Parameter(self.weight.new_zeros((r, r)))

            # Scaling factor: alpha / r for consistency with LoRA-like scaling
            self.scaling = self.kron_alpha / float(self.r)

            # Freeze the pre-trained (original) weight
            self.weight.requires_grad = False

        # Reset parameters of the base layer + KronA factors
        self.reset_parameters()

        # If the original layer is in fan_in_fan_out format, transpose the base weights
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self) -> None:
        """
        Resets base weights and KronA parameters. The base class call
        is identical to nn.Linear.reset_parameters(), but we also
        initialize KronA factors here.
        """
        nn.Linear.reset_parameters(self)
        if hasattr(self, "kronA_A"):
            # Initialize KronA factors. For simplicity, use Kaiming Uniform
            nn.init.kaiming_uniform_(self.kronA_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.kronA_B, a=math.sqrt(5))

    def train(self, mode: bool = True) -> Module:
        """
        The same merging/unmerging logic as LoRA:
          - If we go into train mode, we "unmerge" from the base weight (if merged).
          - If we go into eval mode, we "merge" the KronA update into the base weight
            for faster inference.
        """

        def weight_transposed(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Unmerge the KronA update from self.weight
                if self.r > 0:
                    kron_update = torch.kron(self.kronA_A, self.kronA_B)
                    self.weight.data -= weight_transposed(kron_update) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the KronA update into self.weight
                if self.r > 0:
                    kron_update = torch.kron(self.kronA_A, self.kronA_B)
                    self.weight.data += weight_transposed(kron_update) * self.scaling
                self.merged = True

        return self

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. If KronA is active (r>0) and not merged, apply:

            y = x * W + ( dropout(x) * kronA ) * scaling

        where kronA = kron(kronA_A, kronA_B).
        """

        def transpose(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # Normal forward if KronA not in use or if we've merged it already
        if self.r == 0 or self.merged:
            return F.linear(input, transpose(self.weight), bias=self.bias)

        # Add Kronecker-based adaptation
        #   result = x * W + (dropout(x) @ kron(W_A^T)) * scaling
        base_out = F.linear(input, transpose(self.weight), bias=self.bias)

        kron_update = torch.kron(
            self.kronA_A, self.kronA_B
        )  # shape = [out_features, in_features]
        # dropout(x) has shape [batch, in_features], so we multiply by kron_update^T of shape [in_features, out_features]
        adapted = self.kron_dropout(input) @ kron_update.transpose(0, 1)

        return base_out + adapted * self.scaling


class KronAConv(nn.Module, KronALayer):
    """
    Generic Kronecker-based convolution class to mimic the style of ConvLoRA.
    Expects a `conv_module` such as nn.Conv2d passed in via the constructor.
    Then we flatten the conv.weight to shape (M, N) and define KronA factors
    KronA_A and KronA_B such that kron(KronA_A, KronA_B).shape == (M, N).
    """

    def __init__(
        self,
        conv_module,  # e.g., nn.Conv2d
        in_channels,
        out_channels,
        kernel_size,
        r=0,  # 'r' for Kron factor
        kron_alpha=1,
        kron_dropout=0.0,
        merge_weights=True,
        groups=1,
        **kwargs,
    ):
        super(KronAConv, self).__init__()
        # Create the actual convolution module
        self.conv = conv_module(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            **kwargs,
        )
        # Register the Conv's weight/bias as parameters of *this* module
        # so that we can freeze them if needed
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)

        # Initialize the KronALayer
        KronALayer.__init__(
            self,
            r=r,
            kron_alpha=kron_alpha,
            kron_dropout=kron_dropout,
            merge_weights=merge_weights,
        )

        # We'll handle only the simplest scenario: no dilation or other complexities.
        # (But you can adapt this for more advanced cases.)
        if isinstance(kernel_size, int):
            k = kernel_size
        else:
            # if kernel_size is a tuple (kH, kW), you can adapt similarly.
            # For illustration, just assume it's a square kernel if a tuple:
            k = kernel_size[0]

        # Flatten shape:
        #   conv.weight: [out_channels, in_channels//groups, k, k] => shape MxN with M=out_channels*k^2/groups, N=in_channels*k^2/groups
        self.groups = groups
        M = (out_channels // groups) * (k * k)
        N = in_channels * (k * k) // groups

        if r > 0:
            if (M % r) != 0 or (N % r) != 0:
                raise ValueError(
                    f"For KronAConv, M={M} and N={N} must be multiples of r={r}.\n"
                    "Either choose a different r or adjust in/out channels, kernel_size, or groups."
                )

            # Define KronA parameters so that kron(kronA_A, kronA_B).shape = (M, N)
            self.kronA_A = nn.Parameter(self.conv.weight.new_zeros(M // r, N // r))
            self.kronA_B = nn.Parameter(self.conv.weight.new_zeros(r, r))

            # scaling factor, analogous to LoRA's alpha / r
            self.scaling = float(self.kron_alpha) / float(self.r)
            # Freeze the pre-trained conv.weight
            self.conv.weight.requires_grad = False

        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        """
        Resets the underlying conv module's parameters (like a normal nn.Conv),
        plus a custom reset for KronA factors if they exist.
        """
        self.conv.reset_parameters()
        if hasattr(self, "kronA_A"):
            # Kaiming initialization for both KronA_A, KronA_B by default
            nn.init.kaiming_uniform_(self.kronA_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.kronA_B, a=math.sqrt(5))

    def train(self, mode=True):
        """
        Merge/unmerge logic. If we go into train mode, we "unmerge" KronA from
        self.conv.weight if it was merged. If we switch to eval mode, we can merge
        KronA in for faster inference.
        """
        super(KronAConv, self).train(mode)
        if self.r == 0:
            # No KronA usage
            return self

        if mode:
            # -> Train mode
            if self.merge_weights and self.merged:
                # Un-merge the KronA update from conv.weight
                kron_update = torch.kron(self.kronA_A, self.kronA_B)
                # Reshape to conv.weight's shape
                kron_update = kron_update.view(self.conv.weight.shape)
                self.conv.weight.data -= kron_update * self.scaling
                self.merged = False
        else:
            # -> Eval mode
            if self.merge_weights and not self.merged:
                # Merge KronA into conv.weight
                kron_update = torch.kron(self.kronA_A, self.kronA_B)
                kron_update = kron_update.view(self.conv.weight.shape)
                self.conv.weight.data += kron_update * self.scaling
                self.merged = True

        return self

    def forward(self, input):
        """
        If r>0 and not merged, add the KronA-based update to the frozen conv.weight
        on the fly. Otherwise, just do a normal conv forward.
        """
        if self.r == 0 or self.merged:
            # No KronA or it's merged -> just use self.conv as-is
            return self.conv(input)

        # KronA is active: compute the updated weight = conv.weight + kron(...)*scaling
        kron_update = torch.kron(self.kronA_A, self.kronA_B)  # shape = (M, N)
        kron_update = kron_update.view(self.conv.weight.shape)  # reshape to conv.weight
        adapted_weight = self.conv.weight + kron_update * self.scaling

        return self.conv._conv_forward(input, adapted_weight, self.conv.bias)


class Conv2d(KronAConv):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(KronAConv):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)
