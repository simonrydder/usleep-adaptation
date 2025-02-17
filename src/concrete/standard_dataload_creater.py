from torch.utils.data import DataLoader, Dataset

from src.interfaces.dataload_creater import DataloadCreater


class StandardDataloadCreater(DataloadCreater):
    def __init__(self) -> None:
        super().__init__()

    def create_training_loader(self, train: Dataset) -> DataLoader:
        return DataLoader(train, batch_size=1, shuffle=True)

    def create_validation_loader(self, validation: Dataset) -> DataLoader:
        return DataLoader(validation, shuffle=False)

    def create_test_loader(self, test: Dataset) -> DataLoader:
        return DataLoader(test, shuffle=False)
