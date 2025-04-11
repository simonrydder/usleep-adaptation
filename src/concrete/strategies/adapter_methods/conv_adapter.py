from typing import Literal, Type

import torch
import torch.nn as nn
from lightning import LightningModule
from pydantic import BaseModel

from src.concrete.strategies.forward_passes.sequential_forward_pass import SequentialForwardPass
from csdp.ml_architectures.usleep.usleep import (
    ConvBNELU,
    Decoder,
    Dense,
    Encoder,
    SegmentClassifier,
)
from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.adapter_method import AdapterMethod
from src.interfaces.strategies.forward_pass import ForwardPass


class ConvolutionSetting(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int] | tuple[int, int]
    stride: int | tuple[int] | tuple[int, int]
    padding: int | tuple[int] | tuple[int, int] | str
    dilation: int | tuple[int] | tuple[int, int]
    groups: int
    bias: bool
    ceil_pad: bool = False


class ConvAdapter(AdapterMethod):
    def __init__(
        self,
        forward_pass: ForwardPass,
        activation: Type[nn.Module],
        layer: bool,
        section: Literal["encoder", "decoder", "both"] = "both",
        reduction: int | None = None,
        kernel: int | tuple[int] | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.forward_pass = forward_pass
        self.reduction = reduction
        self.activation = activation
        self.kernel = kernel
        self.layer = layer
        self.section = section

    def apply(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        usleep = getattr(model, "model")
        assert isinstance(usleep, LightningModule) or isinstance(usleep, nn.Module), (
            "Model is not a LightningModule or pytorch Module"
        )

        setattr(model, "model", self.recursive_apply(usleep))

        return model

    def recursive_apply(self, parent: nn.Module | LightningModule) -> nn.Module:
        if isinstance(parent, Dense) or isinstance(parent, SegmentClassifier):
            return parent  # skip segment classifier

        if self.section == "encoder" and isinstance(parent, Decoder):
            return parent  # skip decoder if section is not 'both' or 'decoder'

        if self.section == "decoder" and isinstance(parent, Encoder):
            return parent  # skip encoder if section is not 'both' or 'encoder'

        if self.layer and isinstance(parent, ConvBNELU):
            return self.create_adapter_module(parent)  # type: ignore

        if not self.layer and isinstance(parent, nn.Conv1d):
            return self.create_adapter_module(parent)

        if not self.layer and isinstance(parent, nn.Conv2d):
            return self.create_adapter_module(parent)

        for name, child in parent.named_children():
            setattr(parent, name, self.recursive_apply(child))

        return parent

    def create_adapter_module(self, module: nn.Conv2d | nn.Conv1d) -> nn.Module:
        adapter = self.create_adapter(module)

        return ConvAdapterModule(module, adapter, self.forward_pass)

    def create_adapter(self, module: nn.Conv2d | nn.Conv1d):
        setting = self.create_conv_setting(module)
        conv_type = type(module) if not self.layer else nn.Conv1d
        adapter = ConvAdapterBlock(
            self.reduction, self.kernel, self.activation, conv_type, setting
        )

        return adapter

    def create_conv_setting(self, module: nn.Conv2d | nn.Conv1d) -> ConvolutionSetting:
        in_channels = module.in_channels
        out_channels = module.out_channels
        try:
            # ConvBNELU module
            ceil_pad = getattr(module, 'ceil_pad')
        except AttributeError:
            ceil_pad = False
                    
        if isinstance(self.forward_pass, SequentialForwardPass):
            in_channels = out_channels
        
        return ConvolutionSetting(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=module.kernel_size,  # type: ignore
            stride=module.stride,  # type: ignore
            dilation=module.dilation,  # type: ignore
            padding=module.padding,  # type: ignore
            groups=module.groups,
            bias=module.bias is not None,
            ceil_pad=ceil_pad
        )


class ConvAdapterBlock(nn.Module):
    def __init__(
        self,
        reduction: int | None,
        kernel: int | tuple[int] | tuple[int, int] | None,
        activation_function: Type[nn.Module],
        conv_type: Type[nn.Conv1d] | Type[nn.Conv2d],
        conv_settings: ConvolutionSetting,
    ) -> None:
        super().__init__()
        in_channels = conv_settings.in_channels
        out_channels = conv_settings.out_channels
        self.ceil_pad = conv_settings.ceil_pad
        
        del conv_settings.in_channels
        del conv_settings.out_channels
        del conv_settings.ceil_pad

        if reduction is None:
            reduction = in_channels  # -> hidden dim = 1

        hidden_dim = max(in_channels // reduction, 1)

        if kernel is not None:
            conv_settings.kernel_size = kernel

        # Depth-wise Convolution
        self.W_down = conv_type(in_channels, hidden_dim, **conv_settings.model_dump())

        # Activation function
        self.activation = activation_function()

        # Point-wise Convolution: kernel_size = 1
        self.W_up = conv_type(hidden_dim, out_channels, kernel_size=1)

        # Alpha parameter for scaling output
        alpha_shape = (1, out_channels, 1)
        if conv_type == nn.Conv2d:
            alpha_shape = (*alpha_shape, 1)

        self.alpha_scale = nn.Parameter(torch.ones(alpha_shape), requires_grad=True)
        
        self.ceil_padding = nn.ConstantPad1d(padding=(0, 1), value=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.W_down(x)
        h = self.activation(h)
        h = self.W_up(h)
        
        if self.ceil_pad and h.shape[2] % 2 == 1:
            h = self.ceil_padding(h)

        return h * self.alpha_scale  # Optional if alpha scaling is unnecessary


class ConvAdapterModule(LightningModule):
    def __init__(
        self,
        original: nn.Module,
        adapter: nn.Module,
        forward_pass: ForwardPass,
    ) -> None:
        super().__init__()

        self.original = original
        self.adapter = adapter
        self.forward_pass = forward_pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_pass.forward(
            x, original=self.original, adapter=self.adapter
        )
