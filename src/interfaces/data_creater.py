from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class DataCreater(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_dataloaders(
        self, fold: int, train_size: int | None
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @abstractmethod
    def create_training_loader(self) -> DataLoader:
        """
        Creates a DataLoader for the training dataset.
        """
        pass

    @abstractmethod
    def create_test_loader(self) -> DataLoader:
        """
        Creates a DataLoader for the test dataset.
        """
        pass

    @abstractmethod
    def create_validation_loader(self) -> DataLoader:
        """
        Creates a DataLoader for the validation dataset.
        """
        pass
