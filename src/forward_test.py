import os

import torch

from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.usleep_data_creater import UsleepDataCreater
from src.dataset.resnet.simple_images import SimpleImages
from src.models.conv_simple import ConvSimple
from src.models.usleep import UsleepLightning


def conv_test():
    model = ConvSimple()

    dataloader = StandardDataCreater(
        SimpleImages(10, 1, distribution="shifted")
    ).create_training_loader()

    sample = next(iter(dataloader))

    loss = model.training_step(sample, 0)

    pass

    for name, param in model.named_parameters():
        if "bias" in name:
            continue

        mask = torch.rand(param.shape) > 0.9
        param.register_hook(lambda grad, m=mask: grad * m)

    loss.backward()

    print("After backward:")
    for name, param in model.named_parameters():
        print(
            f"{name}: {param.shape}, Grad: {param.grad.shape if param.grad is not None else None}"
        )
    pass


def usleep_test():
    model = UsleepLightning.load_from_checkpoint(
        os.path.join("data", "ckpt", "usleep", "alternative_big_sleep.ckpt")
    )
    model.to("cpu")

    dataloader = UsleepDataCreater(
        "eesm19.hdf5", (0.8, 0.1, 0.1)
    ).create_training_loader()

    sample = next(iter(dataloader))

    loss = model.training_step(sample, 0)

    masks = {}
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, Grad: {param.grad}")
        mask = torch.rand(param.shape) > -0.9
        masks[name] = mask
        param.register_hook(lambda grad, m=mask: grad * m)

    torch.autograd.set_detect_anomaly(True)
    loss.backward()

    print("After backward:")
    for name, param in model.named_parameters():
        print(
            f"{name}: {param.shape}, Grad: {param.grad.shape if param.grad is not None else None}"
        )
    pass

    for name, param in model.named_parameters():
        new_tensor = param.grad != 0
        new_tensor.sum() / len(new_tensor.flatten())
        print(f"{name}: {new_tensor.sum() / len(new_tensor.flatten())}")


# pass
if __name__ == "__main__":
    usleep_test()
