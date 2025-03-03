import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(tensor):
    return torch.norm(tensor, p=2)


class DoRALayer:
    """
    A base class (mimicking LoRALayer) holding essential hyperparams
    and a learnable scale 'm'. We won't do merges/unmerges here, because
    DoRA is not simply additive anymore.
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool = False,  # We'll ignore 'merge_weights' but keep for API compatibility
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout for the update
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # A learnable scalar magnitude m (initialized later)
        self.m = None


class DoraLinear(nn.Linear, DoRALayer):
    """
    Implements 'DoRA' for a dense (linear) layer:

       1) Base weight W is frozen.
       2) Low-rank update: W_delta = W_B @ W_A (scaled by alpha).
       3) W_cur = W + alpha * W_delta
       4) Normalization: v = W_cur / ||W_cur||
       5) Final weight = m * v  (where m is a learnable scalar)
       6) Forward pass: X @ (m * v)^T + bias

    We preserve 'fan_in_fan_out' if you want the original weight stored as (fan_in, fan_out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,  # We'll ignore merges for DoRA
        bias: bool = True,
        **kwargs,
    ):
        # Init the base nn.Linear
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        # Init DoRALayer fields (r, alpha, dropout, etc.)
        DoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out

        # If r > 0, create the low-rank factors
        if self.r > 0:
            # We do W_B: (out_features, r), W_A: (r, in_features)
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha  # We'll apply / self.r if you prefer
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0

        # Make the base weight frozen
        self.weight.requires_grad = False

        # A learnable scalar for the final magnitude
        self.m = nn.Parameter(torch.tensor(1.0))

        # Reset & possibly transpose if fan_in_fan_out
        self.reset_parameters()
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        # Base initialization
        nn.Linear.reset_parameters(self)

        # Low-rank factors
        if self.lora_A is not None and self.lora_B is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

        # Initialize m to match the base weight's norm (roughly)
        with torch.no_grad():
            base_norm = l2_norm(self.weight) + 1e-8
            self.m.copy_(base_norm)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        DoRA forward:
          1) W_base = (maybe transposed) self.weight (frozen)
          2) W_delta = lora_B @ lora_A (scaled by self.scaling)
          3) W_cur = W_base + scaling * W_delta
          4) v = W_cur / ||W_cur||
          5) W_dora = m * v
          6) output = x @ W_dora^T + bias
        """

        # Helper to handle fan_in_fan_out
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        W_base = T(self.weight)

        if self.lora_A is not None and self.lora_B is not None:
            W_delta = self.lora_B @ self.lora_A
        else:
            W_delta = torch.zeros_like(W_base)

        W_cur = W_base + self.scaling * W_delta

        norm_val = torch.norm(W_cur, p=2) + 1e-8
        W_dora = self.m * (W_cur / norm_val)
        out = F.linear(input, W_dora, self.bias)
        return out
