from lightning import Trainer

from src.concrete.standard_fine_tuner import StandardFineTuner
from src.models.simple import Simple


def main():
    trainer = Trainer()
    ft = StandardFineTuner(trainer)

    model = Simple()


if __name__ == "__main__":
    pass
