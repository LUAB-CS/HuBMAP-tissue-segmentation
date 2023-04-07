import os
import numpy as np
from data_processing.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


def get_training_datasets_and_dataloaders(
    batch_size: int = 4,
    input_size: int = 1024,
    root_dir: str = os.path.join('..','data')
) -> tuple[
    CustomDataset,
    CustomDataset,
    DataLoader,
    DataLoader
]:
    """
    Load the training and test datasets into data loaders.
    """

    transform = None

    total_dataset = CustomDataset(root_dir = root_dir, reshape_size = 1024, transform=transform)
    train_dataset, validation_dataset = random_split(total_dataset,[0.8,0.2])

    if batch_size > 1:
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dl = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        return train_dataset, validation_dataset, train_dl, validation_dl

    return train_dataset, validation_dataset, None, None