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

data_dir = 'data'

image_dir = os.path.join(data_dir,'train_images')
label_dir = os.path.join(data_dir, 'train_annotation')
meta_path = os.path.join(data_dir,'train.csv')
mask_dir = os.path.join(data_dir, 'train_mask')
resized_dir = os.path.join(data_dir,'resized_images')

meta_df = pd.read_csv(meta_path).sort_values(by = 'id')

class CustomDataset(Dataset):
    def __init__(self, root_dir = data_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(os.path.join(root_dir, "train_images"))
        self.label_files = os.listdir(os.path.join(root_dir, "train_mask"))
        self.meta_df = pd.read_csv(os.path.join(root_dir,'train.csv')).sort_values(by = 'id')


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "train_images", self.image_files[idx])
        label_path = os.path.join(self.root_dir, "train_mask", self.label_files[idx])
        image = tifffile.imread(image_path)
        image_tensor = torch.tensor(image)
        label = tifffile.imread(label_path)
        label_tensor = torch.tensor(label)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
            label_tensor = self.transform(label_tensor)
            return (image_tensor, label_tensor)
        return (image_tensor, label_tensor)

train_dataset = CustomDataset()

