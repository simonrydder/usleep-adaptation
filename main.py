from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.concrete.standard_model_updater import StandardModelUpdater
from src.config.config import load_config
from src.dataset.resnet.simple_images import SimpleImages


def fine_tune_model():
    config = load_config("usleep/conv_adapter")

    model_cls = config.get_model_class()
    loader = StandardModelLoader(model_cls)
    old_model = loader.load_pretrained(config.ckpt)

    adapter = config.adapter.get_adapter()
    updater = StandardModelUpdater(adapter)
    new_model = updater.adapt(old_model)

    # print(summarize_lightning(new_model, 2))

    data_creater = StandardDataCreater(SimpleImages(1000, 10, distribution="shifted"))
    # data_creater = StandardDataCreater(SimpleLinear(1000, distribution=2))
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


if __name__ == "__main__":
    fine_tune_model()
