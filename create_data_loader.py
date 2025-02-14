from torch.utils.data import DataLoader

def create_hdf5_dataloader(
    folder_path,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    transform=None
):
    """
    Args:
        folder_path (str): Path to the .hdf5 folder.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Whether to shuffle the data each epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        transform (callable, optional): Optional transform/pipeline to apply.

    Returns:
        DataLoader: a PyTorch DataLoader for the HDF5SleepDataset.
    """
    dataset = HDF5SleepDataset(folder_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # often helps performance if you have enough RAM
    )
    return dataloader
