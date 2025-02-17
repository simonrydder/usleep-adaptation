from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.concrete.standard_model_updater import StandardModelUpdater
from src.config.config import load_config
from src.dataset.resnet.simple_images import SimpleImages
from src.dataset.simple.simple_linear import SimpleLinear


def fine_tune_model():
    config = load_config("simple_bitfit")

    model_cls = config.get_model_class()
    loader = StandardModelLoader(model_cls)
    old_model = loader.load_pretrained(config.ckpt)

    adapter = config.get_adapter()
    updater = StandardModelUpdater(adapter)
    new_model = updater.adapt(old_model)

    data_creater = StandardDataCreater(SimpleImages(1000, 10, distribution="shifted"))
    data_creater = StandardDataCreater(SimpleLinear(1000, distribution=2))
    train = data_creater.create_training_loader()
    val = data_creater.create_validation_loader()
    test = data_creater.create_test_loader()

    model_trainer = StandardModelTrainer()
    trainer = model_trainer.get()

    trainer.test(old_model, test)
    trainer.fit(new_model, train, val)
    trainer.test(new_model, test)

    if config.model == "simple":
        print("Pretrained:", old_model.state_dict())
        print("Fine Tunes:", new_model.state_dict())

    pass


if __name__ == "__main__":
    fine_tune_model()
