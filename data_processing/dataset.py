import os
import torch
import pandas as pd
import tifffile
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, root_dir, reshape_size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.reshape_size = reshape_size
        self.image_files = os.listdir(os.path.join(root_dir, "train_images"))
        self.label_files = os.listdir(os.path.join(root_dir, "train_masks"))
        self.meta_df = pd.read_csv(os.path.join(
            root_dir, 'train.csv')).sort_values(by='id')
        self.encoder = OneHotEncoder().fit(
            np.array(self.meta_df["organ"]).reshape(-1, 1))
        ids = self.meta_df["id"]
        self.encoded_organs = {}
        self.encoded_organs_array = self.encoder.transform(
            np.array(self.meta_df["organ"]).reshape(-1, 1)).toarray()
        for k, id_ in enumerate(ids):
            self.encoded_organs[id_] = torch.tensor(
                self.encoded_organs_array[k])
        self.format_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((reshape_size, reshape_size))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.root_dir, "train_images", self.image_files[idx])
        label_path = os.path.join(
            self.root_dir, "train_masks", self.label_files[idx])
        image = tifffile.imread(image_path)
        label = tifffile.imread(label_path)
        organ = self.encoded_organs[int(self.image_files[idx][:-5])]
        image_tensor = self.format_transform(image)
        label_tensor = self.format_transform(label)
        image_size = self.meta_df[self.meta_df["id"] == int(self.image_files[idx][:-5])]["img_height"].values[0]
        ratio = image_size / self.reshape_size  #ex : 512/3000 -> bigger picture, smaller pixel ?
        pixel_size = torch.tensor(
                ratio * self.meta_df[self.meta_df["id"] == int(self.image_files[idx][:-5])]["pixel_size"].values[0]).type(torch.float32)
        if self.transform is not None:
            (image_tensor, label_tensor) = self.transform(
                (image_tensor, label_tensor))
            return (image_tensor, organ, label_tensor, pixel_size)

        return (image_tensor, organ, label_tensor, pixel_size)


class DebugCustomDataset(Dataset):
    def __init__(self, root_dir, reshape_size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.reshape_size = reshape_size
        self.image_files = os.listdir(
            os.path.join(root_dir, "debug_train_images"))
        self.label_files = os.listdir(
            os.path.join(root_dir, "debug_train_masks"))
        self.meta_df = pd.read_csv(os.path.join(
            root_dir, 'train.csv')).sort_values(by='id')
        self.encoder = OneHotEncoder().fit(
            np.array(self.meta_df["organ"]).reshape(-1, 1))
        ids = self.meta_df["id"]
        self.encoded_organs = {}
        self.encoded_organs_array = self.encoder.transform(
            np.array(self.meta_df["organ"]).reshape(-1, 1)).toarray()
        for k, id_ in enumerate(ids):
            self.encoded_organs[id_] = torch.tensor(
                self.encoded_organs_array[k])
        self.format_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((reshape_size, reshape_size))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.root_dir, "debug_train_images", self.image_files[idx])
        label_path = os.path.join(
            self.root_dir, "debug_train_masks", self.label_files[idx])
        image = tifffile.imread(image_path)
        label = tifffile.imread(label_path)
        organ = self.encoded_organs[int(self.image_files[idx][:-5])]
        image_tensor = self.format_transform(image)
        label_tensor = self.format_transform(label)
        image_size = self.meta_df[self.meta_df["id"] == int(self.image_files[idx][:-5])]["img_height"].values[0]
        ratio = image_size / self.reshape_size  #ex : 512/3000 -> bigger picture, smaller pixel ?
        pixel_size = torch.tensor(
            ratio * self.meta_df[self.meta_df["id"] == int(self.image_files[idx][:-5])]["pixel_size"].values[0]).type(torch.float32)
        if self.transform is not None:
            (image_tensor, label_tensor) = self.transform(
                (image_tensor, label_tensor))
            return (image_tensor, organ, label_tensor, pixel_size)

        return (image_tensor, organ, label_tensor, pixel_size)
