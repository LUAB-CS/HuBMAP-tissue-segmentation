import os
import numpy as np
from data_processing.dataset import CustomDataset, DebugCustomDataset
from data_processing.augmentation import RandomBlur, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, CustomColorJitter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
from torch import Generator


def get_training_datasets_and_dataloaders(
    batch_size: int = 4,
    input_size: int = 1024,
    root_dir: str = os.path.join('..', 'data'),
    mode: str = "normal"  # or debug
) -> tuple[
    CustomDataset,
    CustomDataset,
    DataLoader,
    str
]:
    """
    Load the training and test datasets into data loaders.
    """

    transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(),
        CustomColorJitter(),
        RandomBlur()
    ])
    if mode == "debug":
        total_dataset = DebugCustomDataset(
            root_dir=root_dir, reshape_size=input_size, transform=transform)
    else:
        total_dataset = CustomDataset(
            root_dir=root_dir, reshape_size=input_size, transform=transform)
    encoder = total_dataset.encoder
    generator = Generator().manual_seed(0)
    train_dataset, validation_dataset = random_split(
        total_dataset, [0.8, 0.2], generator=generator)
    validation_dataset.dataset.transform = None
    if batch_size > 1:
        train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        validation_dl = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True)
        return train_dataset, validation_dataset, train_dl, validation_dl, encoder

    return train_dataset, validation_dataset, None, None, encoder
