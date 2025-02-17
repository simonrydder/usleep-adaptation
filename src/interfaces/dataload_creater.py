from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset


class DataloadCreater(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create_training_loader(self, train: Dataset) -> DataLoader:
        pass

    @abstractmethod
    def create_test_loader(self, test: Dataset) -> DataLoader:
        pass

    @abstractmethod
    def create_validation_loader(self, validation: Dataset) -> DataLoader:
        pass
