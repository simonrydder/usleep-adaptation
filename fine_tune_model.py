from src.concrete.standard_adapter import StandardAdapter
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.config.config import load_config


def fine_tune_model(config_file: str):
    config = load_config(config_file)

    model_cls = config.get_model_class()
    loader = StandardModelLoader(model_cls)
    old_model = loader.load_pretrained(config.ckpt)

    adapter_method = config.adapter.get_adapter_method()
    adapter = StandardAdapter(adapter_method)
    new_model = adapter.adapt(old_model)

    data_creater = config.data.get_data_creater()
    train = data_creater.create_training_loader()
    val = data_creater.create_validation_loader()
    test = data_creater.create_test_loader()

    model_trainer = StandardModelTrainer(config.trainer)
    trainer = model_trainer.get()

    trainer.test(old_model, test)
    trainer.fit(new_model, train, val)
    trainer.test(new_model, test)
