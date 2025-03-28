from typing import Any, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader

# from src.concrete.strategies.adapter_methods.utils.fisher import FisherMask
from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.adapter_method import AdapterMethod


class FISH(AdapterMethod):
    def __init__(
        self,
        keep_ratio: float,
        num_samples: int = 1,
        grad_type: Literal["absolute", "square"] = "square",
    ) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.grad_type = grad_type
        self.keep_ratio = keep_ratio

    def apply(
        self,
        model: FrameworkModel,
        **kwargs,
    ) -> FrameworkModel:
        dataloader = kwargs.get("dataloader")
        assert dataloader is not None, "Dataloader is required for FISH adapter method"

        for param in model.parameters():
            param.requires_grad = True

        fish = LabeledFisherMask(
            model,
            dataloader,
            self.num_samples,
            self.keep_ratio,
            self.grad_type,
        )

        masks = fish.calculate_mask()
        model_device = model.device
        print("Model device: ", model_device)
        print("Mask device: ", list(masks.values())[0].device)
        for name, param in model.named_parameters():
            mask = masks[name]
            free_count = (mask == 1).sum().item()
            frozen_count = (mask == 0).sum().item()
            assert free_count + frozen_count == param.numel()
            setattr(param, "free_count", free_count)
            setattr(param, "frozen_count", frozen_count)
            param.register_hook(lambda grad, m=mask: grad * m.to(grad.device))

        return model


class LabeledFisherMask:
    def __init__(
        self,
        model: FrameworkModel,
        train: DataLoader,
        num_samples: int,
        keep_ratio: float,
        grad_type: str,
    ) -> None:
        self.model = model
        self.train_dataloader = train
        self.num_samples = num_samples
        self.keep_ratio = keep_ratio
        self.grad_type = grad_type

    def calculate_mask(self) -> dict[str, Tensor]:
        org_device = self.model.device
        # org_device = list(self.model.parameters())[0].device

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)

        gradients = get_gradients(
            self.model,
            self.train_dataloader,
            self.num_samples,
            self.grad_type,
            device,
        )

        mask: dict[str, Tensor] = {}
        shapes: dict[str, torch.Size] = {}
        non_classifier_tensors = []
        classifier_count: int = 0
        param_count: int = 0
        for name, grad in gradients.items():
            # Classifier layer should be trainable.
            if self.model.is_classification_parameter(name):
                classifier_count += grad.numel()
                mask[name] = torch.ones_like(grad).to(org_device)
            else:
                shapes[name] = grad.shape
                non_classifier_tensors.append(grad.view(-1))

            param_count += grad.numel()

        tensors = torch.concat(non_classifier_tensors, 0)

        keep_num = int(param_count * self.keep_ratio) - classifier_count

        assert keep_num > 0

        top_pos = torch.topk(tensors, keep_num)[1]

        tensor_masks = torch.zeros_like(tensors, device=device)
        tensor_masks[top_pos] = 1

        assert tensor_masks.long().sum() == len(top_pos)

        idx_from = 0
        for name, shape in shapes.items():
            idx_to = idx_from + shape.numel()
            mask[name] = tensor_masks[idx_from:idx_to].reshape(shape).to(org_device)
            idx_from = idx_to

        assert idx_from == len(tensor_masks)

        self.model.to(org_device)

        print(f"Total parameters: {param_count}")
        print(f"Trainable classifier parameters: {classifier_count}")
        print(f"Trainable model parameters: {keep_num}")
        print(
            f"Trainable parameters: {(keep_num + classifier_count) / param_count * 100:.4f} %"
        )

        return mask


def get_gradients(
    model: FrameworkModel,
    data_loader: DataLoader,
    num_samples: int,
    grad_type: str,
    device: str,
) -> dict[str, Tensor]:
    assert num_samples > 0, "num_samples must be integer geater than 0."

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square
    else:
        raise NotImplementedError(f"Grad type: {grad_type} not implemented")

    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = torch.zeros_like(param).to(device)

    model.to(device)

    for idx, data in enumerate(data_loader):
        data = move_data_to_device(data, device)
        if idx >= num_samples:
            break

        pred, label, _ = model.predict_step(data, idx)
        loss = model.loss(pred, label)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None
            gradients[name] += grad_method(param.grad).data

        model.zero_grad()

    return gradients


def move_data_to_device(data: Any, device: str) -> Any:
    if isinstance(data, Tensor):
        return data.to(device)

    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = move_data_to_device(value, device)

        return data

    return data
