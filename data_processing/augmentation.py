import os
import numpy as np
import pandas as pd
import tifffile
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random as r

#Toutes les augmentations sont visualisables dans data_augmentation_check.ipynb

class RandomFlip(object):

    def __call__(self, item):
        image, mask = item
        if r.random() > 0.5:
            return item
        image = image.flip(dims=0)
        mask = mask.flip(dims=0)
        return (image,mask)
    

class RandomRotation(object):

    def angle(self):
        #Sens de rotation
        return r.uniform(-180,180)
    
    def __call__(self,item):
        image,mask = item
        angle = self.angle()
        image = image.permute(2,0,1)
        image = F.rotate(image,angle=angle)
        image = image.permute(1,2,0)
        mask = F.rotate(mask[None,:],angle)
        return (image,mask.squeeze())



class CustomColorJitter(object):

    def __init__(self,brightness = 0.3, hue = 0.3, saturation = 0.3):
        self.brightness = brightness
        self.hue = hue
        self.saturation = saturation

    def __call__(self,item):
        image, mask = item
        image = image.permute(2,0,1)
        image = transforms.ColorJitter(brightness=self.brightness,hue=self.hue,saturation=self.saturation)(image)
        image = image.permute(1,2,0)
        return (image,mask)


class RandomBlur(object):

    def __init__(self, kernel_size=25, blurred_ratio = 0.2):
        self.kernel_size = kernel_size
        self.blurred_ratio = blurred_ratio

    def __call__(self,item):
        image, mask = item
        if r.random() > self.blurred_ratio:
            return item
        image = image.permute(2,0,1)
        image = F.gaussian_blur(image,kernel_size=self.kernel_size)
        image = image.permute(1,2,0)
        return (image,mask)
    
data_transform = transforms.Compose([
    RandomFlip(),
    RandomRotation(),
    CustomColorJitter(),
    RandomBlur()
])

data_transform((torch.randn(3000,3000,3),torch.randn(3000,3000)))



