import os
import torch
import pandas as pd
import tifffile
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, reshape_size, transform=None, normalize=True):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(os.path.join(root_dir, "train_images"))
        self.label_files = os.listdir(os.path.join(root_dir, "train_masks"))
        self.meta_df = pd.read_csv(os.path.join(root_dir,'train.csv')).sort_values(by = 'id')
        self.format_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((reshape_size, reshape_size))
        ])
        pixels = 0.0
        psum = np.zeros(3).astype(float)
        psum_sq = np.zeros(3).astype(float)

        for im_id in os.listdir(os.path.join(root_dir, "train_images")):
            image_path = os.path.join(self.root_dir, "train_images", im_id)
            img = tifffile.imread(image_path)
            pixels += img.shape[0]*img.shape[1]
            psum += np.sum(img.astype(float), axis=(0,1))
            psum_sq += np.power(img.astype(float), 2).sum(axis=(0,1))
        self.total_mean = psum/pixels
        self.total_std = np.sqrt((psum_sq/pixels) - self.total_mean**2)
        self.normalize = normalize
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "train_images", self.image_files[idx])
        label_path = os.path.join(self.root_dir, "train_masks", self.label_files[idx])
        image = tifffile.imread(image_path)
        label = tifffile.imread(label_path)
        organ = self.meta_df[self.meta_df["id"] == int(self.image_files[idx][:-5])]["organ"].values[0]
        image_tensor = self.format_transform(image)
        label_tensor = self.format_transform(label)
        if self.transform is not None:
            (image_tensor, label_tensor) = self.transform((image_tensor, label_tensor))
            return (image_tensor, organ, label_tensor)
        if self.normalize:
            image_tensor = (image_tensor - self.total_mean)/self.total_std

        return (image_tensor, organ, label_tensor)

class DebugCustomDataset(Dataset):
    def __init__(self, root_dir, reshape_size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(os.path.join(root_dir, "debug_train_images"))
        self.label_files = os.listdir(os.path.join(root_dir, "debug_train_masks"))
        self.meta_df = pd.read_csv(os.path.join(root_dir,'train.csv')).sort_values(by = 'id')
        self.format_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((reshape_size, reshape_size))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "debug_train_images", self.image_files[idx])
        label_path = os.path.join(self.root_dir, "debug_train_masks", self.label_files[idx])
        image = tifffile.imread(image_path)
        label = tifffile.imread(label_path)
        organ = self.meta_df[self.meta_df["id"] == int(self.image_files[idx][:-5])]["organ"].values[0]
        image_tensor = self.format_transform(image)
        label_tensor = self.format_transform(label)
        if self.transform is not None:
            (image_tensor, label_tensor) = self.transform((image_tensor, label_tensor))
            return (image_tensor, organ, label_tensor)

        return (image_tensor, organ, label_tensor)

class CustomTestDataset(Dataset):
    def __init__(self, root_dir, reshape_size):
        self.root_dir = root_dir
        self.image_files = os.listdir(os.path.join(root_dir, "test_images"))
        self.meta_df = pd.read_csv(os.path.join(root_dir,'test.csv')).sort_values(by = 'id')
        self.format_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((reshape_size, reshape_size))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "test_images", self.image_files[idx])
        image_id = self.image_files[idx][:-5]
        image = tifffile.imread(image_path)
        organ = self.meta_df[self.meta_df["id"] == int(self.image_files[idx][:-5])]["organ"].values[0]
        image_tensor = self.format_transform(image)

        return (image_id, image_tensor, organ)





