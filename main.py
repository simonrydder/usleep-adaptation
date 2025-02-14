from src.concrete.standard_fine_tuner import StandardFineTuner
from src.models.simple import Simple
from src.utils.callbacks import early_stopping, timer
from src.utils.trainer import define_trainer


def main():
    trainer = define_trainer(
        1,
        "cpu",
        [early_stopping("loss", 5, "min"), timer()],
    )
    ft = StandardFineTuner(trainer)

    model = Simple()


if __name__ == "__main__":
    pass
