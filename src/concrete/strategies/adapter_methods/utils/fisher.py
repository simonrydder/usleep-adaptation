import torch

# from su_torch_backend import DataLoader
from torch.utils.data import DataLoader

from src.interfaces.framework_model import FrameworkModel

CLASSIFIER_NAME = "classifier"


def calculate_the_importance_label(
    model: FrameworkModel, data_loader: DataLoader, num_samples, cuda_device, grad_type
):
    """
    Args:
        grad_type: (square or absolute)
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square
    else:
        raise NotImplementedError()

    idx = 0

    for inputs in data_loader:
        if idx >= num_samples:
            break

        y_pred = model(inputs)
        y_true = inputs["labels"]

        loss = model.loss(y_pred, y_true)

        loss.backward()

        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data

        model.zero_grad()

        idx += 1

    return gradients_dict


class FisherMask:
    def __init__(self, model, train_dataloader, num_samples, keep_ratio, grad_type):
        self.model = model
        self.train_dataloader = train_dataloader
        self.num_samples = num_samples
        self.keep_ratio = keep_ratio
        self.grad_type = grad_type

        self.mask = None

    def calculate_mask(self):
        model = self.model
        num_samples = self.num_samples
        keep_ratio = self.keep_ratio
        grad_type = self.grad_type

        original_device = list(model.parameters())[0].device
        cuda_device = "cpu" if torch.cuda.is_available() else "cpu"

        model.to(cuda_device)

        data_loader = self.train_dataloader
        gradients = calculate_the_importance_label(
            model, data_loader, num_samples, cuda_device, grad_type
        )

        # add sizes and aggregate tensors
        sizes = {}
        tensors = []

        classifier_size: int = 0
        all_params_size = 0

        classifier_mask_dict = {}

        for k, v in gradients.items():
            # don't count classifier layer, they should be all trainable
            if CLASSIFIER_NAME in k:
                classifier_size += torch.prod(torch.tensor(v.shape)).item()
                classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
            else:
                sizes[k] = v.shape
                tensors.append(v.view(-1))

            all_params_size += torch.prod(torch.tensor(v.shape)).item()

        tensors = torch.cat(tensors, 0)

        keep_num = int(all_params_size * keep_ratio) - classifier_size

        assert keep_num > 0

        top_pos = torch.topk(tensors, keep_num)[1]

        masks = torch.zeros_like(tensors, device=cuda_device)

        masks[top_pos] = 1

        assert masks.long().sum() == len(top_pos)

        mask_dict = {}

        now_idx = 0
        for k, v in sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            mask_dict[k] = masks[now_idx:end_idx].reshape(v).to(original_device)
            now_idx = end_idx

        assert now_idx == len(masks)

        # Add the classifier's mask to mask_dict
        mask_dict.update(classifier_mask_dict)

        model.to(original_device)

        # Print the parameters for checking
        classifier_size = 0
        all_params_size = 0
        pretrain_weight_size = 0

        for k, v in mask_dict.items():
            if CLASSIFIER_NAME in k:
                classifier_size += (v == 1).sum().item()
            else:
                pretrain_weight_size += (v == 1).sum().item()

            all_params_size += torch.prod(torch.tensor(v.shape)).item()

        print(pretrain_weight_size, classifier_size, all_params_size)
        print(
            f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %"
        )

        self.mask = mask_dict.values()

        return mask_dict
