import os
import numpy as np
from data_processing.dataset import CustomDataset, CustomTestDataset
from data_processing.augmentation import RandomBlur, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, CustomColorJitter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms


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

    transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(),
        #CustomColorJitter(),
        RandomBlur()
    ])

    total_dataset = CustomDataset(root_dir = root_dir, reshape_size = input_size, transform=transform)
    train_dataset, validation_dataset = random_split(total_dataset,[0.8,0.2])

    validation_dataset.transfrom = None

    if batch_size > 1:
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dl = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        return train_dataset, validation_dataset, train_dl, validation_dl

    return train_dataset, validation_dataset, None, None

def get_test_dataset_and_dataloader(
    batch_size: int = 4,
    input_size: int = 1024,
    root_dir: str = os.path.join('..','data')
) -> tuple[
    CustomTestDataset,
    DataLoader
]:
    """
    Load the training and test datasets into data loaders.
    """

    test_dataset = CustomTestDataset(root_dir = root_dir, reshape_size = input_size)

    if batch_size > 1:
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return test_dataset, test_dl

    return test_dataset, None
