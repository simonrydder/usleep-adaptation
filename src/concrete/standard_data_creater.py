from torch.utils.data import DataLoader, Dataset, random_split

from src.interfaces.data_creater import DataCreater


class StandardDataCreater(DataCreater):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()

        train, val, test = random_split(dataset, [0.7, 0.15, 0.15])

        self.train: Dataset = train
        self.validation: Dataset = val
        self.test: Dataset = test

    def create_training_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self.train, shuffle=True, **kwargs)

    def create_validation_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self.validation, shuffle=False, **kwargs)

    def create_test_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self.test, shuffle=False, **kwargs)
