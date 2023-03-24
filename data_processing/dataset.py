import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import tifffile
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(os.path.join(root_dir, "train_images"))
        self.label_files = os.listdir(os.path.join(root_dir, "train_masks"))
        self.meta_df = pd.read_csv(os.path.join(root_dir,'train.csv')).sort_values(by = 'id')


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "train_images", self.image_files[idx])
        label_path = os.path.join(self.root_dir, "train_masks", self.label_files[idx])
        image = tifffile.imread(image_path)
        label = tifffile.imread(label_path)
        organ = self.meta_df[self.meta_df["id"] == int(self.image_files[idx][:-5])]["organ"].values[0]
        if self.transform is not None:
            image_tensor = self.transform(image)
            label_tensor = self.transform(label)
            return (image_tensor, label_tensor)
        else:
            image_tensor = torch.tensor(image).T
            label_tensor = torch.tensor(label).T
            return (image_tensor, organ, label_tensor)
