import torch

from src.concrete.standard_fine_tuner import StandardFineTuner
from src.concrete.standard_model_updater import StandardModelUpdater
from src.data.resnet.simple_images import SimpleImages
from src.dataloader.standard_dataloader import StandardDataloader
from src.models.resnet import Resnet
from src.utils.callbacks import early_stopping, timer
from src.utils.trainer import define_trainer


def main():
    trainer = define_trainer(
        5,
        "gpu",
        [early_stopping("train_loss", 50, "min"), timer()],
    )
    ft = StandardFineTuner(trainer)

    num_classes = 10

    model = Resnet(num_classes)

    dataloader_creator = StandardDataloader()

    train_dataset = SimpleImages(
        num_samples=1000, num_classes=num_classes, distribution="normal"
    )
    # Fine-tuning dataset (different dummy data).
    finetune_dataset = SimpleImages(
        num_samples=1000, num_classes=num_classes, distribution="shifted"
    )

    train, val, test = torch.utils.data.random_split(train_dataset, [0.8, 0.1, 0.1])

    train_ft, val_ft, test_ft = torch.utils.data.random_split(
        finetune_dataset, [0.8, 0.1, 0.1]
    )

    train_dataloader = dataloader_creator.create_training_loader(train)
    val_dataloader = dataloader_creator.create_validation_loader(val)
    test_dataloader = dataloader_creator.create_test_loader(test)

    train_dataloader_ft = dataloader_creator.create_training_loader(train_ft)
    val_dataloader_ft = dataloader_creator.create_validation_loader(val_ft)
    test_dataloader_ft = dataloader_creator.create_test_loader(test_ft)

    ft.train(model, train_dataloader, val_dataloader)
    ft.test(model, test_dataloader)
    ft.test(model, test_dataloader_ft)
    st_model_updater = StandardModelUpdater()

    new_model = st_model_updater.adapt(model)

    new_trainer = define_trainer(
        5,
        "gpu",
        [early_stopping("train_loss", 50, "min"), timer()],
    )
    ft = StandardFineTuner(new_trainer)
    ft.train(model, train_dataloader_ft, val_dataloader_ft)
    ft.test(new_model, test_dataloader_ft)


if __name__ == "__main__":
    main()
