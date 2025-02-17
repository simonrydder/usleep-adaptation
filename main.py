from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.concrete.standard_model_updater import StandardModelUpdater
from src.data.resnet.simple_images import SimpleImages
from src.models.resnet import Resnet


def fine_tune_model():
    # loader = StandardModelLoader(Simple)
    loader = StandardModelLoader(Resnet)
    # old_model = loader.load_pretrained("data/ckpt/simple_linear_one.ckpt")
    old_model = loader.load_pretrained("data/ckpt/resnet_pretrained.ckpt")

    updater = StandardModelUpdater()
    new_model = updater.adapt(old_model)

    data_creater = StandardDataCreater(SimpleImages(1000, 10, distribution="shifted"))
    train = data_creater.create_training_loader()
    val = data_creater.create_validation_loader()
    test = data_creater.create_test_loader()

    model_trainer = StandardModelTrainer()
    trainer = model_trainer.get()

    trainer.test(old_model, test)
    trainer.fit(new_model, train, val)
    trainer.test(new_model, test)

    # print("Pretrained:", old_model.state_dict())
    # print("Fine Tunes:", new_model.state_dict())


if __name__ == "__main__":
    fine_tune_model()
